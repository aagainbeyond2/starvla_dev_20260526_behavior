# StarVLA Behavior-Skill OFT vs PI/GR00T 代码 Review

> 训练配置：`train_gt_behavior_skill_full_qwenoft_2b_size_proportional_dropout0.yaml`（OFT）  
> 对照：`train_gt_behavior_skill_full_qwen35_2b_size_proportional_dropout0.yaml`（PI）  
> 数据：`behavior_skill_lerobot_datasets` + `gt_behavior_skill_full`  
> 入口脚本：仓库内等价于 `examples/Behavior-Skill-GT/train_files/training_scripts/train_multinode.sh`（用户环境可能使用 `train_2node_fastio.sh`，**该脚本不在本仓库内，需自行核对是否覆盖 framework / DATA_ROOT**）

---

## 1. 训练链路总览

| 组件 | OFT (QwenOFT) | PI (QwenPI_v3) | GR00T (QwenGR00T) |
|------|---------------|----------------|-------------------|
| VLM | Qwen3.5-2B | Qwen3.5-2B | Qwen3.5-2B（若同配置） |
| Action Head | MLPResNet + L1 | Layerwise DiT + Flow Matching | DiT + Flow Matching |
| 数据 / mix / norm | 与 PI **完全一致** | 同左 | 需单独 yaml，datasets 段应对齐 |
| `include_state` | 未开启 | 未开启 | 未开启 |
| 全局 batch | 16 × GPU 数（2 机 16 卡 = **256**；4 机 32 卡 = 512） | 同左 | 同左 |
| VLM 输入 prompt | **额外追加** 30×`🔍` + `<action>…<action>` 后缀 | 原始 `lang`，无 action placeholder | 同 PI |

---

## 2. Video 帧采样

### 2.1 结论

**不是时序多帧采样**：当前时刻 **3 视角各 1 帧**（head / left_wrist / right_wrist），无历史帧。

### 2.2 `index` → 读帧流程

DataLoader 的 `index` **不是** parquet 行号，而是 mixture 采样的 RNG 种子：

1. **`BehaviorSkillEpochSampler`**：每 epoch 对 mixture 长度做 `randperm`，产出 `index` 序列。
2. **`LeRobotMixtureDataset.sample_step(index)`**：用 `safe_hash(epoch, index, seed)` 固定随机：
   - 选 subtask（`balance_dataset_weights=true` → 按 step 数 size_proportional）
   - 选 trajectory（`balance_trajectory_weights=true` → 长 trajectory 权重更高）
   - 均匀随机 `base_index ∈ [0, traj_len-1]`
3. **`get_step_data(trajectory_id, base_index)`**：读 parquet，按 modality 取数。
4. **Video**（`observation_indices = [0]`）：
   - `step_indices = base_index + 0`
   - 越界 clamp 到 `[0, traj_len-1]`
   - `video_timestamp = parquet["timestamp"][step_indices]`
   - `use_local_videos: true` → 读 subtask 本地切段 mp4（`get_video_path`）
   - **`decord`**：`get_frames_by_timestamps` 做 **timestamp 最近邻** 取帧（非插值）
5. **Transform**：`VideoResize(224)` + ColorJitter → `_pack_sample` 再 PIL resize 224。
6. **输出**：`sample["image"]` = 3 张 224×224 PIL + `lang` + `action[30,23]`。

### 2.3 与 PI/GR00T 关系

**帧采样对 OFT / PI / GR00T 完全相同**（同一 `datasets.vla_data` + `R1ProSkillDataConfig`，`action_indices=list(range(30))`）。换 framework 不改 dataloader。

---

## 3. Loss 形式

### 3.1 QwenOFT — 直接 L1 回归

Prompt 末尾追加 30 个 `🔍` action placeholder token，取最后一层 hidden → MLP → 连续 action。

\[
\mathcal{L}_{OFT} = \frac{1}{BTD}\sum_{b,t,d}\left|\hat{a}_{b,t,d} - a_{b,t,d}\right|
\]

- \(B\)：batch size，\(T=30\)：`action_horizon`，\(D=23\)：`action_dim`
- `nn.L1Loss(reduction='mean')`，**23 维全监督，无 mask**
- 代码：`starVLA/model/framework/VLM4A/QwenOFT.py`

### 3.2 QwenPI_v3 / QwenGR00T — Flow Matching（MSE on velocity）

\[
x_t = (1-t)\,\epsilon + t\,a,\quad v = a - \epsilon,\quad
\mathcal{L}_{FM} = \frac{1}{BTD}\sum_{b,t,d}\left(v_{\theta}(x_t,t,c)_{b,t,d} - v_{b,t,d}\right)^2
\]

- \(\epsilon \sim \mathcal{N}(0,I)\)，\(t\) 从 Beta 分布采样
- PI：`LayerwiseFM_ActionHeader`（每层 VLM hidden cross-attn + `project_layers` 压缩到 `action_dit_hidden_dim=1024`）
- GR00T：`GR00T_ActionHeader`（最后一层 VLM hidden cross-attn）
- **训练 loss 同样无 action 维度 mask**

### 3.3 Label 预处理（三者相同）

1. **`b1k_delta`**：仅 `torso / left_arm / right_arm` 前几维减当前 proprio；base、gripper 保持绝对量。
2. **`BehaviorQ99PerTimeTransform`**：per-timestamp q01/q99 线性归一化到 \([-1,1]\)  
   `clamp((x - q01) / (q99 - q01 + 1e-6) * 2 - 1, -1, 1)`

---

## 4. OFT 性能差于 PI/GR00T — 四点核查

### 4.1 L1 vs MSE —— 次要，非主因

| 对比 | 说明 |
|------|------|
| 实际差异 | OFT 是 **direct regression**；PI/GR00T 是 **flow matching**，目标函数不同 |
| 若仅 L1→MSE（仍 direct） | 通常只有小幅差别，不足以解释「差很多」 |
| L1 副作用 | 梯度恒 ±1，在 \([-1,1]\) 归一化空间可能收敛更慢，但 PI 用同样 label 且无 mask |

**优先级：低~中**

### 4.2 Action mask / gripper —— 可排除

| 框架 | 训练 loss 是否 mask |
|------|---------------------|
| OFT | 否，全 23 维 L1 |
| PI | 否，全维 MSE on velocity |
| GR00T | 否，全维 MSE on velocity |

- `generate_action_mask_for_used_keys` / `_patch_behavior_skill_action_mask`：**仅用于 eval 反归一化**（gripper 维 `mask=False` 表示不做连续 denorm），**不参与训练 loss**。
- 仅有 `ABot_M0` / `AML_ActionHeader` 支持 `action_mask` loss，Behavior PI/GR00T 未使用。

**优先级：可排除**

### 4.3 帧采样 —— 可排除

OFT 与 PI yaml 的 `datasets.vla_data` 段一致，同一 dataloader、同一 `observation_indices=[0]`、同一 augment。

**优先级：可排除**

### 4.4 `repeated_diffusion_steps=8` —— 有影响（中等）

| 框架 | 行为 |
|------|------|
| PI | 读 `trainer.repeated_diffusion_steps=8`（`QwenPI_v3.py` L305–307）；VLM forward 1 次，action head 在 **8× batch** 上各采样 \((\epsilon,t)\) |
| GR00T | 读 `framework.action_model.repeated_diffusion_steps`（`QwenGR00T.py` L199–203）；Behavior PI yaml **未写此字段时默认 4**，与 PI 的 8 不一致 |
| OFT | **无** repeat；每 step 仅 1 次 L1 |

等效于 PI 每 optimizer step 对 action head 有 **8 倍 flow 监督密度**（VLM 计算不 ×8）。

**优先级：中**

---

## 5. 其他可能原因（按影响排序）

### （A）Head 架构与 30-step 时序建模 —— **高**

| | OFT | PI / GR00T |
|--|-----|------------|
| Head 参数量 | 2-block MLPResNet，**~42M**（`hidden_dim=4096=2×2048`，非「百万级」但仍远小于 DiT） | DiT + cross-attn + `project_layers`（2B 下仍 **~10²M 量级**，4B 文档 ~538M） |
| 30 步关系 | `reshape(B×30, H)` **逐步独立** MLP；**30 步共享同一套 MLP 权重** | DiT self-attn + **`add_pos_embed=True`** 显式 step 位置编码 |
| step 间交互 | **无** action head 内 self-attention | DiT 块内 self-attn 建模 chunk 内时序 |
| VLM 条件 | 仅 30 个 `🔍` 位置 hidden | PI：每层 VLM hidden cross-attn；GR00T：整段 VLM cross-attn |

Behavior `action_horizon=30, action_dim=23` 一步回归 690 标量；OFT 把「第几步」完全寄托于 VLM 位置编码 + 各 `🔍` hidden 的差异，对长 horizon 明显弱于 DiT。

### （B）推理 one-shot vs 多步 denoise —— **高（sim eval 必查）**

| | OFT | PI / GR00T |
|--|-----|------------|
| 推理 | 1 次 VLM forward + MLP | 默认 `num_inference_timesteps=4` Euler denoise（`LayerwiseFM_ActionHeader.predict_action`） |
| 能力 | 无 iterative refine | 从噪声逐步 refine chunk |

若对比 **sim 成功率** 而非 train loss，差距会被放大。部署侧 `policy_wrapper.py` 对三者走同一 `predict_action` → denorm 路径，**sim 端无额外 OFT 特判**。

### （C）`🔍` action token 条件机制 —— **中~高**

- 30 个 `🔍` **共用同一 token id**（`action_token * chunk_len` 拼接）；逐步区分仅靠 **序列位置** 与 VLM RoPE，无独立 step embedding。
- 30 个 query 在 user prompt 末尾，**DiT 会对全部 VLM token（含 vision patch）做 cross-attn**；OFT 仅读这 30 个位置的 hidden，条件信息路径更窄。
- OFT 额外追加 `" Please predict the next 30 robot actions: <action>…<action>."`，**改变 VLM 对 vision token 的 attention 分布**（更长 text → softmax 竞争）；PI/GR00T **不追加** 此 suffix，二者 VLM 前向输入 **不对称**。
- 需在训练环境确认：`tokenizer("🔍")` 是否为 **单 token**，且 `30×🔍` 是否产出 **恰好 30 个 id**（否则 `_gather_action_token_embeddings` 会 `RuntimeError`；若能正常训练则此项大概率 OK）。

### （D）VLM 梯度路径 —— **中**

- OFT：loss → MLP → 30 个 `🔍` hidden → VLM；梯度在序列末段 **稀疏** 回传。
- PI：DiT cross-attn 对 **所有 VLM token** 有梯度；另加 `project_layers`（每层 LayerNorm+Linear）也参与训练。
- 三者 `learning_rate`：`qwen_vl_interface=1e-5`，`action_model=1e-4` 相同，但 OFT 的 action 监督 **更难有效更新 VLM**。

### （E）训练内 eval 指标 —— **低（仅 tensorboard，且 PI 也不吃 ddim 参数）**

`train_starvla.py` 的 `eval_action_model` 统一传 `use_ddim=True, num_ddim_steps=20`：

- **OFT**：`predict_action` 忽略 kwargs，一步回归。
- **PI/GR00T**：`predict_action` 同样 **不使用** `use_ddim`；实际走 `num_inference_timesteps=4` 的 flow sampler。

因此 `mse_score` / `eval/action_l1_*` **不能公平横向对比** OFT vs PI，但 **不影响 sim**。PI 的 `action_loss_per_dim` 训练日志可用于定位难学维度；OFT **无** per-dim 训练指标。

### （F）action_model LR 相同但 head 规模差很大 —— **中**

三者均为 `action_model: 1e-4`，但 MLP ~42M vs DiT ~10²M+，最优 LR / 正则可能不同；PI Behavior yaml 中 `dropout=0`，OFT MLP 无 dropout 参数。

### （G）state / future_tokens —— **低（Behavior 配置下）**

Behavior 均未开 `include_state`；GR00T `state_encoder` 同样闲置。PI/GR00T 的 `num_target_vision_tokens=32` 为轻微架构差异。

### （H）训练脚本 / 配置陷阱 —— **中（跑错实验时影响极大）**

| 风险点 | 说明 |
|--------|------|
| `DATA_ROOT` 默认值 | `train_multinode.sh` L53 默认 `…/Behavior_Skill_V1.0/easy`，**非全量**；yaml 注释要求 `DATA_ROOT=…/Behavior_Skill_V1.0`。未 export 则只训 easy 子集。 |
| `framework.name` 覆盖 | 早期脚本曾 **硬编码** `--framework.name QwenPI_v3`（注释 L63–64 已说明并删除）；若 `train_2node_fastio.sh` 仍有类似覆盖，OFT yaml 会 **静默跑成 PI**。 |
| `FRAMEWORK` env | 非空时会 `--framework.name` 覆盖 yaml。 |
| 2 机 vs 4 机 | 2×8=16 卡 → 全局 batch **256**；与 yaml 注释「4 机 512」不同，对比旧 PI run 时需对齐。 |
| `train_2node_fastio.sh` | **不在本仓库**；99/96 两机 launch 时需确认：config 路径、NODE_RANK、`DATA_ROOT`、是否误覆盖 framework。 |

---

## 6. OFT 是否对 action 分 bin？

**不会。** Action 全程 **连续 float**：

- 数据侧：Q99 连续归一化（非 binning）。
- 模型侧：MLP 直接输出 `action_dim` 连续值。
- `🔍` token：hidden state 锚点，**不是**离散 action token CE 训练。

唯一 bin 化：`add_discretized_state_to_instruction` 对 **proprio state** 做 256-bin（π₀.5 风格），需 `include_state: true`；Behavior 当前 **未开启**。

---

## 7. 综合结论

| 因素 | 是否解释「差很多」 |
|------|-------------------|
| Head 容量 + 无 horizon 时序建模（共享 MLP、无 step self-attn） | **最可疑** |
| 推理 denoise vs one-shot | **sim 评测必查** |
| VLM 条件注入（emoji 末段 vs 全序列 cross-attn；OFT 额外 prompt 改变 attention） | **可疑** |
| Flow matching + repeated_diffusion_steps | **中等** |
| L1 vs MSE 范数 | **次要** |
| 帧采样差异 | **否** |
| 训练 loss mask/gripper | **否** |
| 脚本/DataRoot/framework 跑错 | **否（若配置正确）；是（若静默跑错）** |

**一句话**：差距主因几乎不是 L1/MSE 或 mask/采样，而是 **OFT = 共享 MLP 一步回归 + 弱/不对称 VLM 条件** vs **PI/GR00T = Flow Matching + DiT + 强 cross-attn + 多步推理 + 更高 action 监督密度**。

---

## 8. 建议 Ablation

| 实验 | 验证 |
|------|------|
| OFT 改 `nn.MSELoss()` direct regression | L1 单独贡献 |
| PI 设 `num_inference_timesteps=1` 再 sim eval | denoise 对 sim 贡献 |
| OFT 给 MLP head 加 `SinusoidalPositionalEncoding`（逐步 query） | 位置编码是否瓶颈 |
| 对齐 `action_loss_per_dim`（PI 已有，OFT 无） | 哪几维拖后腿 |
| 同 VLM/数据，仅换 `framework.name` → GR00T | head 架构是否主因 |
| 核对 GR00T 实际 `repeated_diffusion_steps`（yaml 写 8 vs 代码默认 4） | 配置一致性 |
| 核对训练 log 中 `framework.name`、实际 `DATA_ROOT`、全局 batch | 排除跑错实验 |
| tokenizer 统计：`len(tokenizer("🔍"*30))` 是否 = 30 | emoji placeholder 是否正常 |

---

## 9. 关键代码路径

| topic | 路径 |
|--------|------|
| OFT forward / L1 | `starVLA/model/framework/VLM4A/QwenOFT.py` |
| PI forward / flow | `starVLA/model/framework/VLM4A/QwenPI_v3.py` |
| GR00T forward / flow | `starVLA/model/framework/VLM4A/QwenGR00T.py` |
| Flow loss 实现 | `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`, `GR00T_ActionHeader.py` |
| MLP head | `starVLA/model/modules/action_model/MLP_ActionHeader.py` |
| VLM 输入构建 | `starVLA/model/modules/vlm/QWen3_5.py` (`build_qwenvl_inputs`, `add_generation_prompt=True`) |
| 数据配置 | `examples/Behavior-Skill-GT/train_files/data_registry/data_config.py` |
| Mixture 采样 | `starVLA/dataloader/gr00t_lerobot/datasets.py` (`sample_step`, `get_video`) |
| 读帧 | `starVLA/dataloader/gr00t_lerobot/video.py` (`get_frames_by_timestamps`) |
| Action 归一化 | `starVLA/dataloader/gr00t_lerobot/transform/behavior_state_action.py` |
| 训练 loop / eval | `starVLA/training/train_starvla.py` |
| 多机脚本 | `examples/Behavior-Skill-GT/train_files/training_scripts/train_multinode.sh` |
| Sim 部署 | `deployment/model_server/policy_wrapper.py` |
| OFT yaml | `examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_skill_full_qwenoft_2b_size_proportional_dropout0.yaml` |
| PI yaml | `examples/Behavior-Skill-GT/train_files/training_config/train_gt_behavior_skill_full_qwen35_2b_size_proportional_dropout0.yaml` |

---

## 10. 上线前 Checklist（2 机训练）

- [ ] 两节点 launch 的 config 分别为 OFT / PI yaml，**无** `--framework.name` 误覆盖
- [ ] `DATA_ROOT=…/Behavior_Skill_V1.0`（非 `/easy`）
- [ ] 训练 log 打印 `framework.name: QwenOFT`（或 QwenPI_v3）
- [ ] 全局 batch = `16 × 16 = 256`（2 机）与 PI 对照 run 一致
- [ ] sim eval 时 PI/GR00T 记录 `num_inference_timesteps`；对比 OFT one-shot 时需注明推理模式差异
- [ ] 不用 tensorboard `mse_score` 横向对比 OFT vs PI

---

*Review 日期：2026-07-23（初版 + 二次 code review 合并）*
