#!/usr/bin/env python3
"""单文件版: 为 Behavior_Skill 数据集每个 skill_subtask 叶子生成 meta/tasks.jsonl + meta/modality.json。

零依赖(纯 stdlib), modality.json 内容直接内嵌, 适合 scp 到任意机器单独跑。

支持两种 data_root:
  三级布局: <root>/<难度>/<skill_type>/<skill_subtask>/   (如 Behavior_Skill_V1.0, 一次跑完 easy/normal/hard)
  二级布局: <root>/<skill_type>/<skill_subtask>/          (如 Behavior_Skill_V1.0_v4/easy)
叶子的判定标准是存在 meta/episodes.jsonl; skill_type/skill_subtask 取叶子的上级/本级文件夹名。

tasks.jsonl(v2.0): 每 episode 一行 {"task_index": <episode_index>, "task": "<prompt>"}
  loader 端 modality.json 把 annotation 的 original_key 设为 episode_index,
  用每帧 episode_index 去 tasks.jsonl(按 task_index 建索引)查 prompt, 故 task_index = episode_index。

prompt(文件夹名下划线转空格):
  full         -> "Task Skill Type: {skill_type}. Task Prompt: {skill_subtask}."
  subtask_only -> "Task Prompt: {skill_subtask}."
两种格式写同一文件, 切换即覆盖, 对比实验需顺序跑。

用法:
  python3 gen_meta_standalone.py /path/to/Behavior_Skill_V1.0                 # 全量生成
  python3 gen_meta_standalone.py /path/to/Behavior_Skill_V1.0 --dry-run      # 只打印
  python3 gen_meta_standalone.py /path/to/V1.0_v4/easy --prompt-format subtask_only
"""
import argparse, json
from pathlib import Path

MODALITY = {
    "action": {
        "base":          {"start": 0,  "end": 3,  "original_key": "action"},
        "torso":         {"start": 3,  "end": 7,  "original_key": "action"},
        "left_arm":      {"start": 7,  "end": 14, "original_key": "action"},
        "left_gripper":  {"start": 14, "end": 15, "original_key": "action"},
        "right_arm":     {"start": 15, "end": 22, "original_key": "action"},
        "right_gripper": {"start": 22, "end": 23, "original_key": "action"},
    },
    "state": {
        "joint_qpos_sin": {"start": 34, "end": 56, "original_key": "observation.state"},
        "joint_qpos_cos": {"start": 62, "end": 84, "original_key": "observation.state"},
    },
    "video": {
        "head":        {"original_key": "observation.images.rgb.head"},
        "left_wrist":  {"original_key": "observation.images.rgb.left_wrist"},
        "right_wrist": {"original_key": "observation.images.rgb.right_wrist"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "episode_index"},
    },
}


def iter_leaves(data_root: Path):
    """同时匹配二级与三级布局, 按路径排序输出 (skill_type, skill_subtask, 叶子目录)。"""
    eps = sorted(set(data_root.glob("*/*/meta/episodes.jsonl")) | set(data_root.glob("*/*/*/meta/episodes.jsonl")))
    for ep in eps:
        leaf = ep.parent.parent
        yield leaf.parent.name, leaf.name, leaf


def build_prompt(skill_type: str, skill_subtask: str, prompt_format: str) -> str:
    st = skill_type.replace("_", " ")
    ss = skill_subtask.replace("_", " ")
    if prompt_format == "subtask_only":
        return f"Task Prompt: {ss}."
    return f"Task Skill Type: {st}. Task Prompt: {ss}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--prompt-format", choices=["full", "subtask_only"], default="full",
                    help="full=带 skill_type 前缀; subtask_only=只有 Task Prompt")
    ap.add_argument("--no-modality", action="store_true", help="只写 tasks.jsonl, 不写 modality.json")
    ap.add_argument("--dry-run", action="store_true", help="只打印, 不写文件")
    args = ap.parse_args()

    n_sub = n_ep = 0
    for skill_type, skill_subtask, d in iter_leaves(args.data_root):
        prompt = build_prompt(skill_type, skill_subtask, args.prompt_format)
        eps = []
        with open(d / "meta" / "episodes.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    eps.append(json.loads(line)["episode_index"])
        n_sub += 1
        n_ep += len(eps)
        rel = d.relative_to(args.data_root)
        print(f"[{rel}] {len(eps)} eps | {prompt}")
        if not args.dry_run:
            with open(d / "meta" / "tasks.jsonl", "w") as f:
                for ei in eps:
                    f.write(json.dumps({"task_index": ei, "task": prompt}, ensure_ascii=False) + "\n")
            if not args.no_modality:
                with open(d / "meta" / "modality.json", "w") as f:
                    json.dump(MODALITY, f, indent=2)
                    f.write("\n")
    print(f"\n{'DRY-RUN ' if args.dry_run else ''}[{args.prompt_format}] done: {n_sub} skill_subtasks, {n_ep} episodes")


if __name__ == "__main__":
    main()
