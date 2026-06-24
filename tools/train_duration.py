#!/usr/bin/env python3
"""train_duration.py — 从 TensorBoard event 文件算每个 run 的真实训练时长。

纯标准库,无需 tensorflow/tensorboard:自己读 TFRecord,抽 Event proto 的
field1(wall_time, double)和 field2(step, varint),用 末-首 wall_time 算跨度。

用法:
  python3 train_duration.py [ROOT]        # ROOT 默认当前目录
  python3 train_duration.py results/important_results

run 的划分: ROOT 下第一层子目录名(其内任意深度的 *tfevents* 都归到这个 run)。
注意: 断点续训若写同一目录,跨度会把中间挂起的空闲时间也算进去(见输出的"连续段"提示)。
"""
import struct, glob, os, sys, datetime, collections


def parse_event_file(path):
    """返回 [(wall_time, step), ...]。容错:坏记录跳过。"""
    out = []
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return out
    i, n = 0, len(data)
    while i + 12 <= n:
        ln = struct.unpack("<Q", data[i:i + 8])[0]
        i += 12  # 8B length + 4B length_crc
        if i + ln + 4 > n:
            break
        rec = data[i:i + ln]
        i += ln + 4  # data + 4B data_crc
        p, wt, step = 0, None, None
        try:
            while p < len(rec):
                tag = rec[p]; p += 1
                field, wire = tag >> 3, tag & 7
                if wire == 1:                      # 64-bit
                    if field == 1:
                        wt = struct.unpack("<d", rec[p:p + 8])[0]
                    p += 8
                elif wire == 0:                    # varint
                    shift = val = 0
                    while True:
                        b = rec[p]; p += 1
                        val |= (b & 0x7f) << shift
                        if not (b & 0x80):
                            break
                        shift += 7
                    if field == 2:
                        step = val
                elif wire == 2:                    # length-delimited -> skip
                    shift = ln2 = 0
                    while True:
                        b = rec[p]; p += 1
                        ln2 |= (b & 0x7f) << shift
                        if not (b & 0x80):
                            break
                        shift += 7
                    p += ln2
                elif wire == 5:                    # 32-bit
                    p += 4
                else:
                    break
        except Exception:
            pass
        if wt is not None:
            out.append((wt, step))
    return out


def fmt(sec):
    return str(datetime.timedelta(seconds=int(sec)))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    root = os.path.abspath(root)
    evs = glob.glob(os.path.join(root, "**", "*tfevents*"), recursive=True)
    if not evs:
        print(f"在 {root} 下没找到 *tfevents* 文件"); return
    # 按 ROOT 下第一层目录名归组
    runs = collections.defaultdict(list)
    for ev in evs:
        rel = os.path.relpath(ev, root)
        run = rel.split(os.sep)[0] if os.sep in rel else "(root)"
        runs[run].append(ev)

    print("=" * 78)
    print(f"  训练时长汇总  root={root}")
    print("=" * 78)
    grand_t0 = grand_t1 = None
    for run in sorted(runs):
        recs = []
        for ev in runs[run]:
            recs += parse_event_file(ev)
        wts = [w for w, _ in recs]
        steps = [s for _, s in recs if s and s > 0]
        if not wts:
            print(f"\n### {run}: 无可解析 wall_time"); continue
        t0, t1 = min(wts), max(wts)
        grand_t0 = t0 if grand_t0 is None else min(grand_t0, t0)
        grand_t1 = t1 if grand_t1 is None else max(grand_t1, t1)
        span = t1 - t0
        sps = span / (max(steps) - min(steps)) if len(steps) > 1 and max(steps) > min(steps) else None
        print(f"\n### {run}")
        print(f"  event 文件: {len(runs[run])}   含wall_time记录: {len(wts)}")
        print(f"  开始 : {datetime.datetime.fromtimestamp(t0)}")
        print(f"  结束 : {datetime.datetime.fromtimestamp(t1)}")
        print(f"  跨度 : {fmt(span)}  ({span/3600:.2f} h)")
        if steps:
            print(f"  step : {min(steps)} → {max(steps)}" + (f"   ≈ {sps:.2f} s/step" if sps else ""))
    if grand_t0 is not None:
        print("\n" + "-" * 78)
        print(f"  全部 run 合并跨度(墙钟): {fmt(grand_t1 - grand_t0)}  ({(grand_t1-grand_t0)/3600:.2f} h)")


if __name__ == "__main__":
    main()
