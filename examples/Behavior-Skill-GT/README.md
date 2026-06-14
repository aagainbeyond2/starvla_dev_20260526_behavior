# 配置: train_files/training_config/   启动脚本: train_files/training_scripts/{single_nodes,multi_nodes}/
#       两个配置共用同一套入口脚本, 用 CONFIG_YAML 切换。

# ========== 单机多卡(默认 8 卡) ==========
# 1. 对比试验: 等概率 vs 按数据量
# 主实验(等概率 subtask_uniform, 即默认):
NUM_PROCESSES=8 bash examples/Behavior-Skill-GT/train_files/training_scripts/single_nodes/run_gt_behavior_skill_easy_singlenode.sh
# 对照(按数据量 size_proportional):
CONFIG_YAML=examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_skill_easy_qwen35_08b_size_proportional.yaml \
  NUM_PROCESSES=8 bash examples/Behavior-Skill-GT/train_files/training_scripts/single_nodes/run_gt_behavior_skill_easy_singlenode.sh

# ========== K8s 多机多卡 ==========
# 每个 pod 各执行一次; 平台注入 PET_NNODES/PET_NPROC_PER_NODE/PET_NODE_RANK/PET_MASTER_ADDR/PET_MASTER_PORT。
# 默认(subtask_uniform):
bash examples/Behavior-Skill-GT/train_files/training_scripts/multi_nodes/run_gt_behavior_skill_easy_multinode.sh
# size_proportional 同样用 CONFIG_YAML 切:
CONFIG_YAML=examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_skill_easy_qwen35_08b_size_proportional.yaml \
  bash examples/Behavior-Skill-GT/train_files/training_scripts/multi_nodes/run_gt_behavior_skill_easy_multinode.sh
