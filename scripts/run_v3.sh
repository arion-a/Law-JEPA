cd /root/gas; P=/venv/main/bin/python
$P gen_dataset.py --out data3 --pre_gap 20 --n_pre 3 --workers 8 > gen3.log 2>&1
for s in 0 1 2; do nohup $P train.py --seed $s --data data3 --steps 60000 --aux_w 1.0 --aux_s_w 1.0 --tag _v3 --no_ledger > tr_v3_seed$s.log 2>&1 & done
$P probes.py --mode controls --data data3 --tag _v3 > ctrl_v3.log 2>&1
