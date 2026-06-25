#!/usr/bin/env python3
"""prebuild_steps_index.py — 并行预构建全量 471 subtask 的 steps_data_index.pkl 缓存。

【为什么】全量训练首启时 loader 会对每个 subtask 串行(rank0)读 parquet 建步索引,
471 个 ~5h,期间其余 GPU 100% 自旋空等。本脚本把这步**离线 + 多进程**做掉:
  - 复用训练**同一条** make_behavior_skill_dataset 创建口径(data_cfg 从同一份 yaml 加载,
    delete_pause_frame / modality / use_local_videos 全一致)-> 建出的 .pkl 与训练逐字节等价;
  - 纯 CPU/IO(强制 CUDA_VISIBLE_DEVICES=""),不占 GPU;
  - multiprocessing 并行(I/O bound, 近线性加速)-> ~5h 压到 ~20-30min;
  - 已建的(.pkl 存在)自动跳过 -> 可断点续建。

【缓存可移植】每份 .pkl 只存 (episode_index, step_index) 整数元组 + 元数据(num_trajectories/
total_steps/delete_pause_frame),零绝对路径。整棵 Behavior_Skill_V1.0 的 471 份 .pkl
(各在 <diff>/<type>/<subtask>/meta/ 下)可直接 U 盘拷进内网同结构 -> 内网训练首启读缓存、
跳过那 5h。前提:内网那份数据是同一份(同 episodes/同顺序),缓存按路径读、不校验内容。

【用法】务必 cd 到子模块根(starVLA 靠 cwd import):
  cd <submodule_root>
  python examples/Behavior-Skill-GT/train_files/data_registry/prebuild_steps_index.py \
    --config examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_skill_full_qwen35_2b_size_proportional_dropout0.yaml \
    --data-root /nfs/.../Behavior_Skill_V1.0 \
    --workers 24
  # 校验某 subtask(删缓存重建并和备份比对): --only easy/pick_up/pick_up_the_tray_from_the_countertop --force
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # 纯 CPU, 不占 GPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

# ---- 每个 worker 的进程内全局(由 initializer 填) ----
_DATA_ROOT = None
_DATA_CFG = None
_DELETE_PAUSE = False
_MAKE = None


def _init_worker(config_yaml: str, data_root: str):
    """spawn 子进程初始化一次:注册 registry + 加载 data_cfg(与训练同一份)。"""
    global _DATA_ROOT, _DATA_CFG, _DELETE_PAUSE, _MAKE
    from omegaconf import OmegaConf
    from starVLA.dataloader.gr00t_lerobot import registry
    registry.discover_and_merge()                          # 注册 R1ProSkill + gt_behavior_skill_full
    from starVLA.dataloader.behavior_skill_lerobot_datasets import make_behavior_skill_dataset

    cfg = OmegaConf.load(config_yaml)
    dc = cfg.datasets.vla_data
    OmegaConf.set_struct(dc, False)
    dc.data_root_dir = data_root                            # 与 train.sh 的 DATA_ROOT 覆盖一致(指父目录)
    _DATA_ROOT = data_root
    _DATA_CFG = dc
    _DELETE_PAUSE = bool(dc.get("delete_pause_frame", False))
    _MAKE = make_behavior_skill_dataset


def _build_one(payload):
    """建一个 subtask 的步索引缓存。返回 (d_name, status, n_steps_or_msg)。"""
    d_name, robot_type, force = payload
    pkl = Path(_DATA_ROOT) / d_name / "meta" / "steps_data_index.pkl"
    if pkl.exists() and not force:
        return (d_name, "SKIP", -1)
    try:
        ds = _MAKE(Path(_DATA_ROOT), d_name, robot_type,
                   delete_pause_frame=_DELETE_PAUSE, data_cfg=_DATA_CFG)
        # 触发已发生在 __init__ 里;取步数仅为回报
        n = -1
        for attr in ("all_steps", "_all_steps"):
            if hasattr(ds, attr):
                n = len(getattr(ds, attr))
                break
        return (d_name, "OK", n)
    except Exception as e:
        import traceback
        return (d_name, "ERR", f"{e.__class__.__name__}: {e} | {traceback.format_exc()[-300:]}")


def _load_mixture(config_yaml: str, data_root: str):
    """主进程侧:加载 data_cfg + 取 mix(去重, 与 get_vla_dataset 同口径)。"""
    from omegaconf import OmegaConf
    from starVLA.dataloader.gr00t_lerobot import registry
    registry.discover_and_merge()
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

    cfg = OmegaConf.load(config_yaml)
    dc = cfg.datasets.vla_data
    data_mix = dc.data_mix
    spec = DATASET_NAMED_MIXTURES[data_mix]
    seen, items = set(), []
    for d_name, _w, robot_type in spec:                    # 去重(同 get_vla_dataset)
        key = (d_name, robot_type)
        if key in seen:
            continue
        seen.add(key)
        items.append((d_name, robot_type))
    return data_mix, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="训练 yaml(取 datasets.vla_data 口径)")
    ap.add_argument("--data-root", required=True, help="Behavior_Skill_V1.0 父目录(mix name 三级)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", default=None, help="只建这些 subtask(逗号分隔三级名), 用于校验")
    ap.add_argument("--force", action="store_true", help="即使缓存存在也重建")
    args = ap.parse_args()

    data_mix, items = _load_mixture(args.config, args.data_root)
    if args.only:
        only = set(s.strip() for s in args.only.split(",") if s.strip())
        items = [it for it in items if it[0] in only]
    total = len(items)
    print(f"[prebuild] mix={data_mix} 共 {total} 个 subtask  workers={args.workers}  "
          f"data_root={args.data_root}  force={args.force}", flush=True)

    payloads = [(d_name, robot_type, args.force) for d_name, robot_type in items]
    t0 = time.time()
    ok = skip = err = 0
    errors = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(args.config, args.data_root)) as pool:
        for i, (d_name, status, info) in enumerate(
                pool.imap_unordered(_build_one, payloads, chunksize=1), 1):
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                err += 1
                errors.append((d_name, info))
                print(f"[prebuild][ERR] {d_name}: {info}", flush=True)
            if i % 10 == 0 or i == total:
                el = time.time() - t0
                rate = i / el if el > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"[prebuild] {i}/{total}  ok={ok} skip={skip} err={err}  "
                      f"{el:.0f}s  ETA {eta:.0f}s", flush=True)

    print(f"\n[prebuild] DONE  ok={ok} skip={skip} err={err}  用时 {time.time()-t0:.0f}s", flush=True)
    if errors:
        print(f"[prebuild] {len(errors)} 个失败:", flush=True)
        for d, m in errors[:20]:
            print(f"   - {d}: {m[:160]}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
