# Behavior-Skill 文件清单

## 修改了哪些文件
- `starVLA/training/train_starvla.py`
- `starVLA/dataloader/__init__.py`
- `starVLA/dataloader/gr00t_lerobot/schema.py`
- `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- `starVLA/model/framework/VLM4A/QwenPI_v3.py`
- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `starVLA/training/trainer_utils/trainer_tools.py`

## 新增了哪些文件
- `examples/Behavior-Skill/README.md`
- `examples/Behavior-Skill/behavior1k_norm_stats.json`
- `examples/Behavior-Skill/port_utils.sh`
- `examples/Behavior-Skill/serve_starvla_for_eval.py`
- `examples/Behavior-Skill/start_parallel_eval_behavior_skill.sh`
- `examples/Behavior-Skill/train_files/run_qwenpi_v3_behavior_skill_easy_8gpu.sh`
- `examples/Behavior-Skill/train_files/data_registry/data_config.py`
- `examples/Behavior-Skill/train_files/demo_starvla_behavior_skill_easy_qwen35_08b.yaml`
- `starVLA/dataloader/behavior_skill_lerobot_datasets.py`
- `starVLA/dataloader/gr00t_lerobot/behavior_skill_dataset.py`
- `starVLA/dataloader/gr00t_lerobot/behavior1k_utils.py`
- `starVLA/dataloader/gr00t_lerobot/transform/behavior_state_action.py`
- `starVLA/training/trainer_utils/experiment_logger.py`

## 简要说明
已把 Behavior-Skill 路径收缩到你要的最小功能集：
- behavior dataset
- b1k_delta action
- per-time q99 norm
- 50tasks norm_stats.json
- data augment
- tensorboard
- sampler
