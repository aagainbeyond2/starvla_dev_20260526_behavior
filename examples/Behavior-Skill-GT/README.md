# 1. 对比试验： 等概率和按数据量对比
# 主实验(等概率)：
NUM_PROCESSES=8 bash examples/Behavior-Skill-GT/train_files/run_gt_behavior_skill_easy_8gpu.sh
# 对照(按数据量)：
CONFIG_YAML=examples/Behavior-Skill-GT/train_files/train_gt_behavior_skill_easy_qwen35_08b_size_proportional.yaml \
  NUM_PROCESSES=8 bash examples/Behavior-Skill-GT/train_files/run_gt_behavior_skill_easy_8gpu.sh
