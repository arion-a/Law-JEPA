# Experiment report: do temperature and pressure emerge in a JEPA? (2D gas)

All numbers come from `results/*.json` and `src/ledger/ledger.jsonl` (entry numbers in brackets).

## 1. The question

Train a neural network on **videos of a simulated gas**, with **no labels**. Then ask: does a short list of 8 numbers
inside the network (the "top-level state") contain the gas's **temperature** and **pressure**, in a form a very simple
decoder can read out?

## 2. Terms used in this report

| Term | Meaning |
|---|---|
| Hard-disk gas | N round disks (radius R = 0.003) bouncing in a 2D box. They only interact by colliding like billiard balls. Units: mass = 1, Boltzmann constant = 1. |
| Box, width W, area A | The box has height 1 and width W (1 to 2). The right wall is a movable **piston**. Area A = W x 1. |
| N | Number of disks (128 to 512). |
| T (temperature) | Average kinetic energy per disk: T = 0.5 * mean(vx² + vy²). Faster disks = hotter gas. |
| P (pressure) | Force per length on the walls. Here the **label** is P = N T / A * (1 + 2φ), the ideal gas law with a small correction for disk size. |
| φ (packing fraction) | Fraction of the box area covered by disks, φ = N π R² / A. About 0.001 to 0.015 here. |
| Ideal gas law | P x A = N x T (when φ is tiny). This is the law the experiment asks whether the network "finds". |
| Substep | One simulator time step, DT = 0.0005. |
| Frame | A picture taken every 80 substeps. 32 x 64 pixels. Each pixel shows how many disks sit in it (0 to 3, capped); the piston column is drawn as a bright line. |
| Clip | The group of pictures given to the network at one moment. In the final design: 4 pictures, spaced 20 substeps apart, the last one being the current frame. |
| Event | A scheduled intervention in an episode: **heat** (all speeds multiplied by 0.7 to 1.4 at once) or **piston** (the wall slides to a new width over 25 frames). 0, 1 or 2 events per episode. |
| Episode | One run of the simulator: 300 frames. Dataset: 1000 train, 100 validation, 100 test episodes. |
| Self-supervised | Learning from the data alone, no T or P labels. Labels exist in the files but are used only at the end, for probing. |
| JEPA | Joint-Embedding Predictive Architecture. The network predicts the *internal code* of something it did not see (a hidden part, or the future), not the raw pixels. |
| Encoder | The part that turns pictures into codes. Here a **transformer** (a network that lets all image parts exchange information) with 4 layers. |
| Patch / token / code | Each picture is cut into 8 x 8-pixel **patches** (32 per picture). Each patch becomes a vector of 128 numbers, its **code** (also called token). |
| Level 0 (L0) | Masked prediction: hide 8 of the 32 patches (two 2x2 blocks) and predict their codes for the *next* clip, from the visible ones. |
| Level 1 (L1) | Predict the top-level state 50 frames later from the current state plus the **action** (how much W changes and the total heating factor over those 50 frames). |
| Top-level state s | A pooled summary: the 32 patch codes are averaged and squeezed by a small network into **8 numbers**. This is what we test. |
| EMA target | The "answer key" for the predictions comes from a slow-moving copy of the network (exponential moving average of its weights), a standard trick to prevent the network from learning nothing. |
| Collapse | The failure where the network outputs the same code for everything. We guard against it with VICReg-style terms (keep each code dimension varied and uncorrelated) and a floor on the spread of s. |
| Motion losses (added in v2/v3) | Two extra training targets computed from the pictures themselves: per patch, the amount of change between pictures, and the particle count; and the same for the whole picture, predicted linearly from s. |
| Linear probe | A straight-line decoder (ridge regression) fitted from s to a quantity such as log T. If it predicts well, the information is *linearly readable* in s. |
| Ridge regression | Linear regression with a penalty on weight size; the penalty is chosen on validation data only. |
| R² | Score of a decoder: 1.0 = perfect, 0 = no better than guessing the average, negative = worse. Always reported on the **test** episodes. |
| Controls | Two decoders run to check the test is meaningful: **pixel control** (decode from the raw pixels, 8192 values for 4 frames) and **untrained control** (decode from the same network before any training). |
| Seed | The random-number starting point of a training run. 3 seeds show the result is not luck of one run. |
| Decision rule | Fixed before any result: SUPPORTED only if, in all 3 seeds, the trained network's R² >= 0.90 for both log T and log P, and both controls score at least 0.30 lower. |

## 3. What was done, step by step

**Step 1: simulator.** A numba-compiled hard-disk simulator moves disks, bounces them off walls, collides pairs, and
moves the piston. Check passed: P x A / (N T) matched 1 + 2φ within 3% in 3 test cases and energy drift stayed below 2% [#1-2].

**Step 2: dataset.** 1200 episodes x 300 frames, generated in 253 s. T always stayed within 0.3 to 2.0 [#3-4].

**Step 3: model, probes, controls.** HJEPA (1.3M weights) written and tested on CPU. Controls ran: neither the pixel
control nor the untrained network reached 0.90 on log T or log P, so the test is not trivially passable [#5-6].

**Step 4: first training (v1).** A rented GPU (vast.ai RTX 4090) trained 3 seeds x 60,000 steps (47 min per seed). Training was
healthy: no collapse, validation L0 loss fell from 0.73 to 0.040 [#7-10]. Design used: 2-frame clips, frames spaced 80 substeps.

**Step 5a: result of v1.** Rule applied: **NOT SUPPORTED** [#11-12]. Diagnostic: log T was not readable at any layer
(R² <= 0.00), and log P reached only about 0.53. Simple hand-built features (frame differences and particle counts)
gave log T 0.40 and log P 0.71, so the information was in the pictures but the network discarded it [#13].

**Step 5b: why.** Two causes found [#14-17]:
1. *Frame spacing.* Temperature is read from how far disks move between pictures. With 80 substeps they move too far to be matched. A study across spacings gave log T readable at R² 0.70 / 0.84 / 0.90 / 0.83 / 0.38 / 0.02 for spacing 5 / 10 / 20 / 40 / 80 / 160. Spacing 20 is best.
2. *No incentive.* The level-0 loss barely changed when motion was removed (0.0439 vs 0.0447), so the network had no reason to keep motion information.
Short 15,000-step tests showed both fixes were needed: spacing 20 **and** a motion loss on both the patch codes and the state s.

**Step 5c: v2 (two repairs).** Spacing 20, plus the two motion losses. Result: log T 0.78 to 0.80, log P 0.90: **NOT SUPPORTED**
(log T too low) [#18-19]. A feature study predicted that 4 pictures instead of 2 would lift log T to about 0.96 [#20].

**Step 5d: v3 (final design).** 4-frame clips, spacing 20, both motion losses, 3 seeds x 60,000 steps, same decision rule.
Training ran at 0.058 s/step with no collapse or NaN. Rule applied: **SUPPORTED** [#21-22].

## 4. Outcome (v3, test episodes, linear probe R²)

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Trained network, log T | 0.929 | 0.943 | 0.944 |
| Trained network, log P | 0.966 | 0.972 | 0.973 |
| Pixel control, log T / log P | -0.021 / 0.483 | same | same |
| Untrained network, log T / log P | -0.005 / 0.052 | -0.008 / 0.072 | 0.001 / 0.094 |

Sanity targets: log N 0.998 to 0.999 and log A 0.988 to 0.994 (trained). Largest control score: 0.483, so the gap is at least 0.48.
Progression of the main number (log T from s): v1 about 0.00, v2 0.78 to 0.80, v3 0.93 to 0.94.

## 5. What this does and does not show (read before citing)

- **It shows** that an 8-number state, trained without T or P labels, linearly encodes log T and log P above 0.90 in all 3 seeds, with controls far lower.
- **The design was repaired after failures.** The rule stayed fixed, but the *setup* changed twice after seeing probe results on the same test episodes (v1 -> v2 -> v3). Treat v3 as a result after tuning, not a first-shot prediction. A clean confirmation needs a new, untouched test set or the 3D experiment.
- **Not unaided emergence.** The motion losses are targets we built from picture differences and counts. They push the network to keep motion and particle count. Hand-built frame-difference features alone give log T 0.77 and log P 0.88 with 2 frames, and about 0.96 predicted with 4 frames, so the network's 0.93 to 0.94 sits close to what we engineered in.
- **Log P is partly inherited.** The pressure label is a formula of N, T and W. N and W are easy to read (0.99), so a good log P follows once log T is decent. The pixel control still gets only 0.48 on log P, so the learning does matter.
- **Scope:** one simulator, 2D, linear probes, no interventions test, no other physics.
