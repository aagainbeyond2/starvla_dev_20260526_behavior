#!/usr/bin/env python3
"""verify_consistency.py — 比对"训练串行建"的 steps 缓存 vs "并行脚本重建"的缓存。
用法: python verify_consistency.py <ref_dir> <data_root> <name1> <name2> ...
ref_dir 里放串行真值(文件名 = name 把 / 换成 __, 后缀 .pkl)。
"""
import sys
import pickle
from pathlib import Path

ref_dir = Path(sys.argv[1])
data_root = Path(sys.argv[2])
names = sys.argv[3:]

all_pass = True
for name in names:
    ref = pickle.load(open(ref_dir / (name.replace("/", "__") + ".pkl"), "rb"))
    cur = pickle.load(open(data_root / name / "meta" / "steps_data_index.pkl", "rb"))
    ck = ref["config_key"] == cur["config_key"]
    nt = ref["num_trajectories"] == cur["num_trajectories"]
    ts = ref["total_steps"] == cur["total_steps"]
    st = ref["steps"] == cur["steps"]
    dp = ref["delete_pause_frame"] == cur["delete_pause_frame"]
    ok = ck and nt and ts and st and dp
    all_pass = all_pass and ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: "
          f"config_key={ck} num_traj={nt} total_steps={ts} STEPS_IDENTICAL={st} "
          f"delete_pause={dp}  (n_steps={ref['total_steps']})")

print("\n==> ALL_PASS" if all_pass else "\n==> SOME_FAILED")
sys.exit(0 if all_pass else 1)
