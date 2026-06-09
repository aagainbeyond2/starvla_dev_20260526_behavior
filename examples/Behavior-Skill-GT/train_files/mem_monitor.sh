#!/usr/bin/env bash
# mem_monitor.sh — 监控主机 RAM + train_starvla 进程 RSS, 检测训练期间是否内存泄漏。
#
# 判读: MemAvailable 持续单调下降 / train 进程 RSS 持续上涨 = 泄漏苗头;
#       稳定在一个区间波动 = 正常(只是峰值占用)。
#
# 用法:
#   bash mem_monitor.sh                         # 默认 30s 间隔, 日志 ./results/Checkpoints/mem_monitor.csv
#   bash mem_monitor.sh 15 /path/mem.csv        # 自定义 间隔秒 + 日志路径
#   MEM_MON_PATTERN='train_starvla.py' bash mem_monitor.sh   # 自定义匹配进程
set -u

INTERVAL="${1:-30}"
LOG="${2:-./results/Checkpoints/mem_monitor.csv}"
PATTERN="${MEM_MON_PATTERN:-train_starvla.py}"

mkdir -p "$(dirname "$LOG")"
[ -s "$LOG" ] || echo "ts,mem_total_GB,mem_used_GB,mem_avail_GB,buff_cache_GB,n_train_procs,train_rss_GB,max1_worker_GB" >> "$LOG"

echo "[mem_monitor] interval=${INTERVAL}s pattern='${PATTERN}' log=${LOG} (Ctrl-C 停止)" >&2

while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  # 系统内存(GB), 取自 /proc/meminfo(单位 kB)
  read -r mt ma used bc < <(awk '
    /^MemTotal:/      {t=$2}
    /^MemAvailable:/  {a=$2}
    /^Buffers:/       {b=$2}
    /^Cached:/        {c+=$2}
    /^SReclaimable:/  {c+=$2}
    END{printf "%.1f %.1f %.1f %.1f", t/1048576, a/1048576, (t-a)/1048576, (b+c)/1048576}' /proc/meminfo)
  # train 进程: 个数 / RSS 汇总(GB) / 单进程最大 RSS(GB)
  read -r n rss top < <(ps -eo rss,cmd 2>/dev/null | awk -v p="$PATTERN" '
    index($0,p)>0 && $0 !~ /awk|mem_monitor/ {n++; s+=$1; if($1>m)m=$1}
    END{printf "%d %.2f %.2f", n+0, s/1048576, m/1048576}')
  echo "${ts},${mt},${used},${ma},${bc},${n},${rss},${top}" >> "$LOG"
  printf "[%s] avail=%sG used=%sG buff/cache=%sG | train: %s procs RSS=%sG (max1=%sG)\n" \
    "$ts" "$ma" "$used" "$bc" "$n" "$rss" "$top" >&2
  sleep "$INTERVAL"
done
