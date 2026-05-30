# blackW yaw 姿态与髋关节变形改动记录

日期：2026-05-30  
任务：改善 `blackW` 轮足机器人 yaw 速度跟踪时的姿态变形问题  
涉及文件：

- `legged_gym/legged_gym/envs/blackW/blackW_config.py`
- `legged_gym/legged_gym/envs/blackW/blackW_env.py`

## 1. 背景

前几轮 reward 和 command curriculum 改动后，`blackW` 的 x/y/yaw 速度跟踪问题基本缓解。但在 yaw 速度跟踪，尤其是纯 yaw 指令下，出现了新的姿态问题：

```text
机器人主要依靠轮子差速转动；
四个轮子长期接触地面；
由于四轮在长方形布局下不能像差速小车一样干净原地转向；
机身 yaw 与地面接触约束产生冲突；
冲突主要表现为髋关节外展/内收变形，而不是正常抬脚换支撑。
```

混合指令下策略偶尔会出现抬脚动作，纯 yaw 下则更容易只转轮子、不抬脚，导致身体扭曲着旋转。

## 2. 已尝试但回退的方案

本轮讨论中曾尝试过两个辅助 shaping，后续训练 500 iteration 后观察改进一般，因此已经删除。

### 2.1 yaw 接触轮侧向滑移惩罚

曾新增过 `yaw_contact_lateral_slip`，意图是在纯 yaw 下惩罚接触轮在机身 y 方向的横向侧滑。

后续判断：该项对“不抬脚、髋关节变形”的主问题不够直接，且量级可能偏弱。当前代码中已完全删除：

- 删除 `_reward_yaw_contact_lateral_slip()`
- 删除 `yaw_contact_lateral_slip_deadband`
- 删除 reward scale `yaw_contact_lateral_slip`

### 2.2 yaw 下 Raibert boost

曾在纯 yaw 下提高 Raibert 奖励权重，意图增强 yaw 落脚点引导。

后续判断：Raibert 主要作用在摆动腿落脚规划上，如果策略本身不进入 swing，相当于入口没有打开。因此该项也已删除：

- 删除 `yaw_raibert_scale`
- 删除 `_reward_raibert()` 返回值中的 yaw boost

## 3. 保留的 yaw 轮速追踪衰减

虽然不再使用侧滑惩罚和 Raibert boost，但保留了纯 yaw 下对轮速追踪奖励的衰减。

### 3.1 yaw 主导权重

新增 `_yaw_activity()`：

```text
yaw_activity = command_activity(cmd_yaw)
lin_quiet = clamp((yaw_lin_cmd_threshold - norm(cmd_xy)) / yaw_lin_cmd_threshold, 0, 1)
yaw = yaw_activity * lin_quiet
```

含义：只有 yaw 指令活跃且 x/y 线速度指令很小时，该权重才接近 1。

当前配置：

```python
yaw_lin_cmd_threshold = 0.1
```

### 3.2 轮速 tracking 在 yaw 主导时削弱

`_reward_wheel_vel_ref_tracking()` 保留原 tracking 结构，但增加 yaw 主导衰减：

```text
wheel_tracking_scale = 1 - yaw * (1 - yaw_wheel_tracking_scale)
reward = tracking * move_cmd * wheel_tracking_scale
```

当前配置：

```python
yaw_wheel_tracking_scale = 0.2
```

含义：纯 yaw 下不关闭轮速追踪，但只保留 20% 权重；混合指令或 x 指令下仍基本保持原轮速 tracking 作用。

## 4. 新增 yaw 接触髋关节变形惩罚

本轮最终采用的核心方案是直接针对观察到的髋关节变形，而不是先规定必须抬脚。

### 4.1 设计目标

```text
纯 yaw 时允许髋关节参与摆腿和调整；
但如果轮子/脚仍在接触地面，就不允许髋关节长期大幅偏离默认姿态来吸收硬拧约束。
```

这比直接提高全局 `hip_pos` 更具体，也比强制 all-feet-contact penalty 更柔和。

### 4.2 hip 与 foot 接触顺序对齐

在 `_init_blackW_dof_indices()` 中新增：

```python
hip_indices_by_foot
```

它按 `self.feet_names` 中的腿前缀找到对应 hip joint，使得：

```text
contact_forces[:, feet_indices, 2] 的第 i 条腿
与
dof_pos[:, hip_indices_by_foot] 的第 i 个 hip
一一对应。
```

这样接触门控不会因为 Isaac Gym DOF 顺序或 foot 顺序不同而错配。

### 4.3 新 reward：`yaw_contact_hip_deviation`

新增 `_reward_yaw_contact_hip_deviation()`：

```text
yaw = _yaw_activity()
hip_error = abs(hip_pos - default_hip_pos)
excess = relu(hip_error - yaw_hip_deviation_margin)
contact_prob = sigmoid((contact_force_z - 5.0) * 0.5)
reward = yaw * sum(abs(excess) * contact_prob)
```

注意：当前实现使用 `abs(excess)`，也就是超过 margin 后近似 L1 惩罚，而不是平方惩罚。这样对中等变形更直接，梯度不会因为刚超过 margin 而过小。

当前配置：

```python
yaw_hip_deviation_margin = 0.1
yaw_contact_hip_deviation = -10.0
```

含义：纯 yaw 下，接触腿 hip 偏离默认姿态超过 0.1 rad 的部分会被惩罚。

## 5. 当前工作区同时包含的训练配置调整

当前 diff 中还包含一些训练/环境配置调整。这些不是 yaw hip penalty 本身，但会影响本轮实验结果，复盘时需要一起记录。

### 5.1 轮距参数

```python
wheel_base_half_width = 0.20975
```

原值为 `0.183`。

### 5.2 command 采样概率

当前配置：

```python
stand_command_prob = 0.1
x_command_prob = 0.0
y_command_prob = 0.0
yaw_command_prob = 0.0
mixed_command_prob = 0.9
```

注意：虽然新增的 `_yaw_activity()` 主要面向 yaw 主导/纯 yaw 情况，但当前采样分布仍以 mixed command 为主。如果后续要专门验证纯 yaw 抬脚和髋关节变形，应单独调整 command 分布或 play 指令。

### 5.3 地形与自适应 reward

```python
mesh_type = 'trimesh'
terrain_adaptive.enabled = True
```

这会影响 orientation、smoothness、action_rate 等地形自适应缩放项。

### 5.4 foothold 与 entropy

```python
foothold = -5.0
entropy_coef = 0.002
```

`foothold` 原值为 `-1.0`，`entropy_coef` 原值为 `0.003`。

## 6. 验证

语法检查通过：

```bash
python -m py_compile \
  legged_gym/legged_gym/envs/blackW/blackW_config.py \
  legged_gym/legged_gym/envs/blackW/blackW_env.py
```

## 7. 后续观察重点

TensorBoard 重点看：

- `Episode/rew_yaw_contact_hip_deviation`
- `Episode/rew_hip_pos`
- `Episode/rew_wheel_vel_ref_tracking`
- `Episode/rew_tracking_ang_vel`
- `Episode/rew_foothold`
- `Episode/rew_trot`
- `Episode/rew_foot_clearance`
- `Train/mean_reward`

`play` 重点看：

1. 纯 yaw 下髋关节外展/内收变形是否减轻。
2. 纯 yaw 下是否仍然只靠轮子硬拧。
3. 如果髋关节变形减少但 yaw tracking 明显变差，说明惩罚过强或 margin 过小。
4. 如果 yaw tracking 保持但仍不抬脚，后续可能需要引入更直接的 step initiation/contact schedule 类 shaping。

## 8. 风险与调参建议

### 8.1 惩罚过强的风险

当前参数：

```python
yaw_hip_deviation_margin = 0.1
yaw_contact_hip_deviation = -10.0
```

相对偏强。如果出现纯 yaw 无法跟踪、动作僵硬或过早终止，可尝试：

```python
yaw_hip_deviation_margin = 0.15 ~ 0.20
yaw_contact_hip_deviation = -4.0 ~ -8.0
```

### 8.2 惩罚过弱的风险

如果髋关节仍明显变形，可尝试：

```python
yaw_hip_deviation_margin = 0.08 ~ 0.12
yaw_contact_hip_deviation = -10.0 ~ -15.0
```

### 8.3 与抬脚行为的关系

该项不是直接奖励抬脚，而是惩罚“接触状态下的过量 hip 变形”。理想效果是策略为了保持 yaw tracking，同时避免接触腿 hip 变形，会更倾向于释放接触、换支撑或调整步态。

如果训练后只是减少 yaw 速度而不是抬脚，需要继续补充更明确的抬脚启动信号。
