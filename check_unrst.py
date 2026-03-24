import resfo
import numpy as np

def compare_arrays(s_val, r_val, name):
    if len(s_val) != len(r_val):
        print(f"{name} length mismatch: {len(s_val)} vs {len(r_val)}")
        return
    if s_val.dtype.kind in 'if':
        diff_idx = np.where(s_val != r_val)[0]
        if len(diff_idx) > 0:
            print(f"{name} Diff at indices {diff_idx[:10]}")
            for idx in diff_idx[:10]:
                print(f"  [{idx}] Sample={s_val[idx]}, Ref={r_val[idx]}")

ref = list(resfo.read('data/spe1/SPE1CASE1.UNRST'))
sam = list(resfo.read('data/sample/SPE1CASE1.UNRST'))

# We want to check IGRP, IWEL, ICON in Step 1 (record 10 to 30)
for i in range(10, min(len(ref), len(sam))):
    n_r, d_r = ref[i]
    n_s, d_s = sam[i]
    if n_r.strip() in ['IGRP', 'IWEL', 'ICON', 'SGRP', 'SWEL', 'SCON', 'ZWEL', 'ZWLS', 'ZGRP']:
        compare_arrays(d_s, d_r, n_r.strip())
