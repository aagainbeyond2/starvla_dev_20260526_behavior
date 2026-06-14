#!/usr/bin/env bash
# ============================================================================
# K8s 多机多卡训练入口。每个节点(pod)各执行一次本脚本; 由平台注入的环境变量完成 rendezvous。
# 与单机同样用 CONFIG_YAML 切两个配置:
#   默认(subtask_uniform):  bash .../multi_nodes/run_gt_behavior_skill_easy_multinode.sh
#   size_proportional:       CONFIG_YAML=.../training_config/..._size_proportional.yaml bash .../multi_nodes/...
#
# 平台注入的拓扑变量(torch elastic 风格, PET_* 优先, 回退到非 PET_* / torchrun 习惯):
#   PET_NNODES         节点数            (回退 NNODES)
#   PET_NPROC_PER_NODE 每节点进程数=GPU数 (回退 NPROC_PER_NODE)
#   PET_NODE_RANK      本节点 rank 0..N-1 (回退 NODE_RANK; 注意 RANK 是全局 rank, 不能直接当节点 rank)
#   PET_MASTER_ADDR    rendezvous 主机   (回退 MASTER_ADDR)
#   PET_MASTER_PORT    rendezvous 端口   (回退 MASTER_PORT)
# 训练走 accelerate launch + deepspeed(deepspeed_multinode_launcher: standard), 故由每节点
# 各自的 accelerate 以 --machine_rank 静态 rendezvous(与本仓库 robocasa 多机同一模式)。
# ============================================================================
set -euo pipefail

# ---- 从平台环境变量推导多机拓扑 ----
NNODES="${PET_NNODES:-${NNODES:-1}}"
NPROC_PER_NODE="${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
MAIN_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
MAIN_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"
# 总进程数 = 节点数 × 每节点进程数(= 总 GPU 数); 算不出来时用 WORLD_SIZE 兜底。
TOTAL_PROCESSES=$(( NNODES * NPROC_PER_NODE ))
if [[ "${TOTAL_PROCESSES}" -le 0 ]]; then
  TOTAL_PROCESSES="${WORLD_SIZE:-8}"
fi

# ---- 多机网络/NCCL: 不强制 loopback; IB/RoCE/网卡按集群实际情况(下面都是可被覆盖的保守默认)----
# 有 IB/RoCE 的集群应 export NCCL_IB_DISABLE=0; 纯 TCP 集群设 1。需要指定网卡/IB 卡时在平台 export
# NCCL_SOCKET_IFNAME=<网卡名> 和 NCCL_IB_HCA=<卡列表> —— 本脚本故意不强设, 以免覆盖集群正确值。
# 参考官方 Robotwin 多机(run_robotwin_train_batch.sh)的 IB 实践: 多机时列全 mlx5 接口更稳, 例:
#   export NCCL_SOCKET_IFNAME=bond0 ; export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export MASTER_ADDR="${MAIN_ADDR}"
export MASTER_PORT="${MAIN_PORT}"

# ---- 共用主体: cd repo 根 + 解析配置 + 组装 LAUNCH_TAIL[] ----
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_common.sh"

echo "[multinode] NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} NODE_RANK=${NODE_RANK} TOTAL_PROCESSES=${TOTAL_PROCESSES} MASTER=${MAIN_ADDR}:${MAIN_PORT}"
echo "[multinode] CONFIG_YAML=${config_yaml}  data_root=${behavior_skill_data_root}"
echo "[multinode] output_dir=${output_dir}"

accelerate launch \
  --config_file "${accel_ds_config}" \
  --num_machines "${NNODES}" \
  --num_processes "${TOTAL_PROCESSES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MAIN_ADDR}" \
  --main_process_port "${MAIN_PORT}" \
  "${LAUNCH_TAIL[@]}"
