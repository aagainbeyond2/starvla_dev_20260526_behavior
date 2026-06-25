# starVLA 训练启动(Behavior-Skill-GT / easy)

**两个零依赖脚本**搞定全部训练,直接 `bash` 运行,不 source / 不调别的脚本。

| 脚本 | 用途 |
|---|---|
| `train.sh` | 单机多卡(8 卡,常用) |
| `train_multinode.sh` | 多机多卡(K8s,每个 pod 各跑一次) |

---

## 这组实验是什么(2×2 消融)

两个自变量,各 2 个取值,组合成 4 个配置:

**① 模型大小**（config 里的 `base_vlm`）
- `0.8B` = Qwen3.5-0.8B
- `2B`  = Qwen3.5-2B

**② 数据采样方式**（config 里的 `balance_dataset_weights`，决定 34 个 skill_subtask 怎么混采）
- `subtask_uniform`（`=false`）= **各子任务等概率**：每个子任务 1/34，不管它有多少条数据。
- `size_proportional`（`=true`）= **按数据量成正比**：子任务的采样权重 × 它的样本数，数据多的子任务被采到的多。

### 4 个配置

| 短名 | 模型大小 | 数据采样 | 在消融里的角色 | 输出目录(`results/Checkpoints/`) |
|---|---|---|---|---|
| `08b_uniform` | 0.8B | 等概率 uniform | **基线** | `gt_behavior_skill_easy_v1_subtask_uniform` |
| `08b_size` | 0.8B | 按数据量 proportional | 对照基线**只改数据采样方式** | `gt_behavior_skill_easy_v1_size_proportional` |
| `2b_uniform` | 2B | 等概率 uniform | 对照基线**只改模型大小** | `gt_behavior_skill_easy_qwen35_2b_subtask_uniform` |
| `2b_size` | 2B | 按数据量 proportional | **模型大小 + 数据采样方式 都改** | `gt_behavior_skill_easy_qwen35_2b_size_proportional` |

---

## 单机启动(最常用)

```bash
bash train.sh 08b_uniform     # 0.8B + 等概率(基线)
bash train.sh 08b_size        # 0.8B + 按数据量
bash train.sh 2b_uniform      # 2B  + 等概率
bash train.sh 2b_size         # 2B  + 按数据量
```

建议在 tmux 里跑(断 ssh 不掉):

```bash
tmux new -s wgt
bash train.sh 2b_size
# Ctrl-b d 脱离;  tmux a -t wgt 回来
```

> 起训头几行会打印 `[train] output = .../<目录>` 和 `[train] log = .../train_<时间戳>.log` —— **核对 output 含目标 config 名再走开**,防止跑错。

---

## 通用格式:跑任意 config(含你新建的 yaml)

上面 4 个**短名只是快捷方式**。`train.sh` / `train_multinode.sh` 的第一个参数其实接**任意 config**,三种写法都认:

```bash
bash train.sh <短名 | config文件名 | config路径>
```

| 写法 | 例子 | 实际解析为 |
|---|---|---|
| **短名**(4 个内置) | `bash train.sh 2b_size` | `training_config/train_..._2b_size_proportional.yaml` |
| **config 文件名**(放在 `training_config/` 下) | `bash train.sh train_my_new_exp.yaml` | `training_config/train_my_new_exp.yaml` |
| **完整 / 相对路径**(任意位置) | `bash train.sh /abs/path/my.yaml`<br>`bash train.sh some/dir/my.yaml` | 原样用(相对路径相对子模块根) |

**新做了一个 yaml 怎么跑:**
- 丢进 `examples/Behavior-Skill-GT/train_files/training_config/` → `bash train.sh <你的yaml文件名>`
- 或放任意位置 → `bash train.sh <路径>`

输出目录 = 该 yaml 里的 `run_id`(yaml 没写则用文件名当目录名);**所有环境变量照样生效**(`STEPS` / `SAVE_INTERVAL` / `DATA_ROOT` / `RUN_ID` / `RUN_ROOT_DIR` / `NUM_PROCESSES`)。例:

```bash
STEPS=300000 bash train.sh train_my_new_exp.yaml         # 新 yaml + 30万步
RUN_ID=my_run_300k bash train.sh /abs/path/to/my.yaml    # 顺便指定输出子目录名
```

多机同理:`bash train_multinode.sh <短名 | 文件名 | 路径>`。

---

## 多机启动(K8s)

每个节点(pod)各执行**同一条**命令,平台注入拓扑变量完成 rendezvous:

```bash
bash train_multinode.sh <短名>     # 短名同上表
```

平台需注入(torch elastic 风格,`PET_*` 优先,回退非 `PET_*`):

```
PET_NNODES  PET_NPROC_PER_NODE  PET_NODE_RANK  PET_MASTER_ADDR  PET_MASTER_PORT
```

有 IB/RoCE 的集群再加:`export NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=<网卡> NCCL_IB_HCA=<卡列表>`;纯 TCP 集群 `NCCL_IB_DISABLE=1`。

---

## 日志 / 产出在哪

全部落到 **输出目录** `results/Checkpoints/<run_id>/`(上表最后一列):

| 内容 | 路径 |
|---|---|
| **训练过程日志**(进度/INFO/"Checkpoint saved") | `train_<时间戳>.log`(多机:`train_node<rank>_<时间戳>.log`) ← 脚本用 `tee` 自动落盘,同时打屏 |
| 中间 checkpoint | `checkpoints/steps_*_pytorch_model.pt` |
| 最终模型 | `final_model/` |
| TensorBoard 曲线 | `tensorboard/` → 看曲线:`tensorboard --logdir results/Checkpoints/<run_id>/tensorboard` |
| 配置快照 | `config.yaml` / `config.full.yaml`(train_starvla.py 自存) |
| 每步 summary | `summary.jsonl` |

---

## 可选环境变量(两脚本通用)

| env | 作用 | 默认 |
|---|---|---|
| `STEPS` | 训练总步数(覆盖 `trainer.max_train_steps`;LR 余弦衰减跟着拉到该步数) | yaml 的 `150000` |
| `SAVE_INTERVAL` | 存档间隔步数(覆盖 `trainer.save_interval`) | yaml 的 `2500` |
| `NUM_PROCESSES` | 单机进程数(=GPU 数),仅 `train.sh` | `8` |
| `DATA_ROOT` | 训练数据根(覆盖 yaml) | `.../training_data/Behavior_Skill_V1.0/easy` |
| `RUN_ID` | 输出子目录名(覆盖 yaml) | yaml 的 `run_id` |
| `RUN_ROOT_DIR` | 输出根目录(覆盖 yaml) | yaml 的 `run_root_dir`(=`./results/Checkpoints`) |

例:**2B + 按数据量采样 + 30 万步**(存档间隔放大到 5000,避免 ckpt 太多)→
```bash
STEPS=300000 SAVE_INTERVAL=5000 bash train.sh 2b_size
```

---

## 脚本做了什么(无需关心,记录在此)

`train.sh` / `train_multinode.sh` 各自一条龙:
1. 配置短名 → 完整 config 路径(`training_config/` 下);
2. `cd` 到子模块根(yaml/数据/norm_stats 都是相对它的路径);
3. 激活 venv `/opt/venv/starvla-qwen35`;
4. **单机**:清掉 pod 注入的多机 env(`RANK/WORLD_SIZE/PET_*` 等,否则非 master 节点 rank 8-15、NCCL barrier 挂)+ loopback NCCL;**多机**:读 `PET_*` 拓扑 + 静态 `--machine_rank`;
5. 从 yaml 取 `run_id`/`run_root_dir`(env 可覆盖),建输出目录;
6. `accelerate launch` DeepSpeed ZeRO-2 跑 `starVLA/training/train_starvla.py`,`tee` 落盘日志。

> 切配置只换短名,**别把 config 当某个内层脚本的位置参数传**——历史坑:旧的 `single_nodes/run_..._singlenode.sh` 只认 `CONFIG_YAML` 环境变量、会静默无视位置参数、跑默认 0.8B 基线(白跑过一次 91h)。现在这两个扁平脚本直接读短名,不再有这个坑。

---

## 全量数据训练(easy+normal+hard, 471 subtask / 75.4M 帧)

全量配置(2B / size_proportional / dropout0 / `norm_stats_full.json` / 300k steps ≈ 2 epoch @ 4机×8卡):
- **PI**:`train_gt_behavior_skill_full_qwen35_2b_size_proportional_dropout0.yaml`
- **OFT**:`train_gt_behavior_skill_full_qwenoft_2b_size_proportional_dropout0.yaml`(QwenOFT MLP/L1 head)

依赖改动:`data_registry/gen_full_mixture.py` 生成 `gt_behavior_skill_full.json`(471)→ data_config 注册 mix `gt_behavior_skill_full`;`behavior_skill_dataset.py` 的 `_BAD_EPISODES` 跳过坏 parquet `wipe_the_trumpet/episode_3700117`。

**⚠️ 必须把 `DATA_ROOT` 覆盖成父目录**(全量 mix name 是三级 `<diff>/<type>/<subtask>`,入口脚本默认 DATA_ROOT 指 `/easy` 会只跑 easy):
```
DATA_ROOT=/nfs/AIGC/weiguoting/AAAI_VLA_2027/behavior-skill-sim/datasets_training/training_data/Behavior_Skill_V1.0
```

**先单机冒烟**(强烈建议,尤其 OFT 是 QwenOFT+behavior 首次组合):
```bash
DATA_ROOT=.../Behavior_Skill_V1.0 STEPS=200 \
  bash examples/Behavior-Skill-GT/train_files/training_scripts/train.sh \
    train_gt_behavior_skill_full_qwen35_2b_size_proportional_dropout0.yaml   # 或 ..._qwenoft_...
```
看:471 subtask 加载不崩、坏 ep 被跳(warning)、`Loaded external action norm stats from .../norm_stats_full.json`、loss 非 NaN。

**4 机正式训**(每台一个 pod,平台注入 `PET_*`):
```bash
DATA_ROOT=.../Behavior_Skill_V1.0 \
  bash examples/Behavior-Skill-GT/train_files/training_scripts/train_multinode.sh \
    train_gt_behavior_skill_full_qwen35_2b_size_proportional_dropout0.yaml
```
全局 batch = 16 × 32 = 512;OFT 换成 `..._qwenoft_...` 那个 config。
