#!/usr/bin/env python3
"""为 Behavior_Skill_V1.0_v4/easy 的每个 skill_subtask 生成 meta/tasks.jsonl。

布局: <data_root>/<skill_type>/<skill_subtask>/ (7 个 skill_type 下共 34 个 skill_subtask 叶子)
每个叶子是 LeRobot v2.0 数据集, meta/episodes.jsonl 列出 episode_index。

prompt(按文件夹名, 下划线统一转空格):
  full         -> "Task Skill Type: {skill_type}. Task Prompt: {skill_subtask}."
  subtask_only -> "Task Prompt: {skill_subtask}."
  {skill_type}    = skill_type 文件夹名转空格 (close_door -> "close door")
  {skill_subtask} = skill_subtask 文件夹名转空格 (close_the_fridge_door -> "close the fridge door")
两种格式用于对比实验(看 skill_type 前缀是否有用); 注意写到同一 meta/tasks.jsonl,
切换格式即覆盖, 故两组实验顺序跑(各用不同 run_id)。

tasks.jsonl(v2.0): 每 episode 一行 {"task_index": <episode_index>, "task": "<prompt>"}
  loader 端 modality.json 把 annotation 的 original_key 设为 episode_index,
  会用每帧的 episode_index 去 tasks.jsonl(按 task_index 建索引)查 prompt,
  所以这里 task_index = episode_index。
"""
import argparse, json
from pathlib import Path


def iter_skill_subtasks(data_root: Path):
    for skill_type_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for skill_subtask_dir in sorted(p for p in skill_type_dir.iterdir() if p.is_dir()):
            if (skill_subtask_dir / "meta" / "episodes.jsonl").exists():
                yield skill_type_dir.name, skill_subtask_dir.name, skill_subtask_dir


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
    ap.add_argument("--dry-run", action="store_true", help="只打印, 不写文件")
    args = ap.parse_args()

    n_sub = n_ep = 0
    for skill_type, skill_subtask, d in iter_skill_subtasks(args.data_root):
        prompt = build_prompt(skill_type, skill_subtask, args.prompt_format)
        eps = []
        with open(d / "meta" / "episodes.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    eps.append(json.loads(line)["episode_index"])
        n_sub += 1
        n_ep += len(eps)
        print(f"[{skill_type}/{skill_subtask}] {len(eps)} eps | {prompt}")
        if not args.dry_run:
            with open(d / "meta" / "tasks.jsonl", "w") as f:
                for ei in eps:
                    f.write(json.dumps({"task_index": ei, "task": prompt}, ensure_ascii=False) + "\n")
    print(f"\n{'DRY-RUN ' if args.dry_run else ''}[{args.prompt_format}] done: {n_sub} skill_subtasks, {n_ep} episodes")


if __name__ == "__main__":
    main()
