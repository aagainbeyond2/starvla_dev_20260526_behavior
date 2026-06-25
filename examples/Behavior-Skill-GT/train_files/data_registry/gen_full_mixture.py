#!/usr/bin/env python3
"""gen_full_mixture.py — 扫 Behavior_Skill_V1.0/{easy,normal,hard} 生成全量混采清单。

产出同目录的 gt_behavior_skill_full.json(list of [name, weight, robot_type]),
data_config.py 读它注册 mix `gt_behavior_skill_full`(471 subtask)。
name 用三级 `<difficulty>/<skill_type>/<skill_subtask>`(loader 是 dataset_path = data_root/name,
data_root 指 Behavior_Skill_V1.0 父目录)。

坏 episode(hard/wipe/wipe_the_trumpet/.../episode_03700117)【不在这里排除】——整段保留(还有 199 个好 ep),
坏帧在 loader 里按 episode_index 跳过(见 behavior_skill_dataset.py 的 _BAD_EPISODES)。

只认含 data/ 子目录的叶子(=一个 lerobot subtask)。可复现:改了数据重跑即可。

用法: python3 gen_full_mixture.py [DATA_ROOT]
  DATA_ROOT 默认本机 .../Behavior_Skill_V1.0
"""
import json
import os
import sys

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else (
    "/workspace/disk/AAAI_VLA_2027/behavior-skill-sim/datasets_training/"
    "training_data/Behavior_Skill_V1.0"
)
ROBOT = "R1ProSkill"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_behavior_skill_full.json")


def main():
    entries = []
    per = {}
    for diff in ("easy", "normal", "hard"):
        dpath = os.path.join(DATA_ROOT, diff)
        if not os.path.isdir(dpath):
            print(f"[warn] 缺难度目录: {dpath}")
            per[diff] = 0
            continue
        n = 0
        for st in sorted(os.listdir(dpath)):          # skill_type
            stp = os.path.join(dpath, st)
            if not os.path.isdir(stp):
                continue
            for sub in sorted(os.listdir(stp)):       # skill_subtask
                subp = os.path.join(stp, sub)
                if os.path.isdir(os.path.join(subp, "data")):
                    entries.append([f"{diff}/{st}/{sub}", 1.0, ROBOT])
                    n += 1
        per[diff] = n
    json.dump(entries, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"per-difficulty: {per}")
    print(f"total: {len(entries)} subtask -> {OUT}")


if __name__ == "__main__":
    main()
