# blackW 轮足机器人 reward 改动实验记录

日期：2026-05-24  
对象：`blackW` 轮足机器人训练任务  
改动文件：

- `legged_gym/legged_gym/envs/blackW/blackW_config.py`
- `legged_gym/legged_gym/envs/blackW/blackW_env.py`

## 1. 问题背景

在训练 `blackW` 轮足机器人时，观察到一个比较典型的异常现象：

1. 由于轮子较重，机器人在横移和原地旋转时不容易抬腿。
2. 为了让策略产生横移和旋转动作，之前将 `tracking_lin_vel_y` 和 `tracking_ang_vel` 的 reward scale 调得很大。
3. 训练日志中有时 reward 很高，看起来训练结果不错。
4. 但在 `play` 或部署时，策略可能对指令几乎没有响应，机器人始终保持静止或只维持姿态。

该现象说明：训练 reward 的数值并没有稳定反映“策略真实跟踪指令”的能力。策略很可能学到了某种 reward 结构漏洞，而不是学到了可部署的横移/旋转行为。

## 2. 原 reward 结构的主要问题

### 2.1 非活跃指令轴会白拿 tracking 分

原始 `blackW` 中，x、y、yaw 三个方向分别计算正向 tracking reward：

- `tracking_lin_vel`
- `tracking_lin_vel_y`
- `tracking_ang_vel`

其形式近似为：

```text
exp(-(cmd - actual)^2 / sigma)
```

这会带来一个问题：当某个轴的指令为 0，而机器人也保持该轴速度为 0 时，该轴 reward 会直接接近满分。

例如：

- x-only 指令时，y=0 和 yaw=0 会自动获得很高的 y/yaw tracking reward。
- yaw-only 指令时，x=0 和 y=0 会自动获得很高的 x/y tracking reward。
- 若 y/yaw 的 scale 被调得很大，则“保持静止”本身就能拿到大量 reward。

旧配置中：

```python
tracking_lin_vel = 2.0
tracking_lin_vel_y = 15.0
tracking_ang_vel = 20.0
```

这意味着策略可以在不真正执行横移/旋转的情况下，仅靠非活跃轴匹配 0 速度拿到较高回报。

### 2.2 单纯增大 y/yaw tracking scale 会放大漏洞

如果继续增大 `tracking_lin_vel_y` 或 `tracking_ang_vel`，理论上会增强 y/yaw 的目标牵引；但在当前分轴 reward 结构下，它也会同时放大非活跃轴的静止奖励。

因此问题不在于 y/yaw 奖励“不够大”，而在于奖励是否只在对应指令轴活跃时才生效。

### 2.3 yaw-only 指令没有充分利用轮子

原始轮速参考 `_target_wheel_velocities()` 只由 `cmd_x` 产生：

```text
wheel_target = cmd_x / wheel_radius
```

这意味着 yaw 指令下，轮子没有明确的差速目标。原地旋转主要依赖腿部摆动和接触切换，而 `blackW` 的轮子较重，这会显著增加学习难度。

同时原始 `stand_still_wheels` 在 `cmd_x` 约为 0 时惩罚轮速。这会误伤 yaw-only 情况：原地旋转需要左右轮差速，但因为 x=0，轮子运动反而会被静止轮速惩罚压制。

### 2.4 yaw-only 卡死检测不足

继承自 `black` 的 stuck 检测主要根据平移指令和沿平移指令方向的速度判断是否卡死。对于纯 yaw 指令，如果机器人不旋转，原有 stuck 逻辑不一定会将其及时 reset。

这会让训练过程采集大量“yaw 指令下原地不动”的低质量样本，且 episode 可能持续较久。

### 2.5 gait 类正奖励权重偏大

旧配置中：

```python
feet_air_time = 5.0
trot = 10.0
raibert = 2.0
```

这些 reward 本意是辅助抬腿、步态和落脚点规划。但如果权重过大，策略可能优先学习“原地踏步形态”或“接触相位形态”，而不是优先完成速度/角速度指令跟踪。

## 3. 本次改动目标

本次改动的核心目标不是简单增加 reward，而是让 reward 更符合实验目的：

```text
只有响应当前指令轴，才获得主要正奖励。
非活跃轴不再白拿正奖励，反而需要保持安静。
x/yaw 尽量利用轮子完成，y 方向主要通过腿部侧向步态完成。
gait/air time/Raibert 作为辅助约束，不作为主要收入来源。
```

具体目标：

1. 修复非活跃轴白拿 tracking 分的问题。
2. 将 tracking 改为“命令活跃门控 + 相对误差 tracking + 同方向 progress”。
3. 为非活跃轴增加速度惩罚。
4. 让 yaw 指令产生轮子差速参考。
5. 修复 yaw-only 情况下轮速被 `stand_still_wheels` 惩罚的问题。
6. 降低 gait 类正奖励权重，避免原地踏步拿高分。
7. 增加 yaw-only 卡死 reset 辅助训练。

## 4. 具体实现

### 4.1 新增命令活跃度函数

新增 `_command_activity(command)`：

```text
activity = clamp((abs(command) - deadzone) / (full - deadzone), 0, 1)
```

配置参数：

```python
command_activity_deadzone = 0.05
command_activity_full = 0.2
```

含义：

- `abs(cmd) <= 0.05` 时，认为该轴基本无指令。
- `abs(cmd) >= 0.2` 时，认为该轴完全活跃。
- 中间区域平滑过渡，避免硬开关导致 reward 抖动。

### 4.2 tracking 改为相对误差 + progress

新增 `_axis_tracking_progress_reward(command, actual, min_ref)`。

其核心结构：

```text
ref = max(abs(command), min_ref)
relative_error = (command - actual) / ref

tracking = exp(-relative_error^2 / relative_tracking_sigma)
progress = clamp(sign(command) * actual / ref, 0, 1)

reward = activity * (0.6 * tracking + 0.4 * progress)
```

配置参数：

```python
relative_tracking_sigma = 0.25
relative_tracking_min_lin_cmd = 0.2
relative_tracking_min_yaw_cmd = 0.3
tracking_reward_weight = 0.6
progress_reward_weight = 0.4
```

设计理由：

- 绝对误差 tracking 在大指令、大误差时容易接近 0，学习信号较弱。
- progress 项能鼓励策略先朝正确方向动起来。
- tracking 项继续负责精确跟踪。
- activity 门控保证只有该轴有指令时才给正奖励。

修改后的三个任务 reward：

- `_reward_tracking_lin_vel`
- `_reward_tracking_lin_vel_y`
- `_reward_tracking_ang_vel`

均调用同一个 helper，保持结构一致。

### 4.3 新增非活跃轴速度惩罚

新增 `_reward_inactive_axis_vel`，并在 config 中加入：

```python
inactive_axis_vel = -0.5
inactive_lin_vel_weight = 1.0
inactive_ang_vel_weight = 0.25
```

其形式：

```text
penalty =
    (1 - active_x)   * vx^2
  + (1 - active_y)   * vy^2
  + (1 - active_yaw) * 0.25 * wz^2
```

作用：

- x-only 时，约束 y/yaw 不要乱动。
- y-only 时，约束 x/yaw 不要乱动。
- yaw-only 时，约束 x/y 不要乱动。
- stand 指令时，三轴都应尽量保持静止。

这与“活跃轴给正奖励、非活跃轴给惩罚”的设计相匹配。

### 4.4 轮速参考加入 yaw 差速

修改 `_target_wheel_velocities()`。

旧结构：

```text
target = cmd_x / wheel_radius
```

新结构：

```text
target = (cmd_x - cmd_yaw * wheel_side_sign * wheel_base_half_width) / wheel_radius
target = target * wheel_forward_sign
```

配置中已有：

```python
wheel_radius = 0.103
wheel_base_half_width = 0.183
```

设计理由：

- x 指令继续对应四轮同向滚动。
- yaw 指令对应左右轮差速。
- 对轮足机器人而言，yaw 不应完全依赖抬腿转动；轮子差速能显著降低原地旋转难度。

### 4.5 轮速 tracking 从 x-only 扩展到 x/yaw

修改 `_reward_wheel_vel_ref_tracking()`：

- 目标轮速使用新的 `_target_wheel_velocities()`。
- 奖励只在 x 或 yaw 指令活跃时生效。
- 使用相对误差而非固定 `sigma=8.0`。

配置参数：

```python
wheel_tracking_relative_sigma = 0.25
wheel_tracking_min_ref = 0.5
wheel_vel_ref_tracking = 1.0
```

### 4.6 修复 stand_still_wheels 与 yaw-only 的冲突

旧逻辑：

```python
is_still = abs(cmd_x) < 0.1
```

新逻辑：

```python
is_still = norm(cmd_xy) < 0.1 and abs(cmd_yaw) < 0.1
```

设计理由：

- 只有三轴指令都接近 0 时，才应该惩罚轮子转动。
- yaw-only 时，轮子差速是期望行为，不应被静止轮速惩罚。

### 4.7 新增 yaw-only 卡死检测

在 `BlackWEnv.check_termination()` 中扩展 yaw 卡死检测：

```text
yaw_cmd = abs(cmd_yaw) > stuck_command_threshold
yaw_progress = sign(cmd_yaw) * base_ang_vel_z
yaw_stalled = yaw_progress < stuck_yaw_vel_threshold
```

配置参数：

```python
stuck_yaw_vel_threshold = 0.1
```

作用：

- 对纯 yaw 指令，如果机器人长期不朝目标方向旋转，则进入 stuck 计时。
- 达到 `stuck_timeout_s` 后 reset。
- 减少训练中“yaw 指令下长时间静止”的样本。

### 4.8 调整 reward scale

本次将任务主奖励调回接近尺度：

```python
tracking_lin_vel = 2.0
tracking_lin_vel_y = 2.0
tracking_ang_vel = 2.0
wheel_vel_ref_tracking = 1.0
```

同时关闭旧的独立 progress：

```python
progress = 0.0
```

原因：progress 已经被并入每个活跃轴的 tracking-progress reward 中。保留旧 progress 可能造成重复奖励，且旧 progress 只看平移方向，不覆盖 yaw。

降低 gait 类正奖励：

```python
feet_air_time = 0.8
trot = 1.0
raibert = 1.0
```

原因：这些项继续作为辅助 shaping，但不应超过任务跟踪本身。

## 5. 本次代码层面的结果

已完成：

1. 新增命令活跃度门控。
2. 修改 x/y/yaw tracking reward。
3. 新增非活跃轴速度惩罚。
4. 修改轮子目标速度，使 yaw 产生左右轮差速。
5. 修改轮速 tracking，使其覆盖 x/yaw。
6. 修复 yaw-only 下轮速被静止惩罚的问题。
7. 新增 yaw-only 卡死 reset。
8. 降低 y/yaw 超大 tracking scale 和 gait 类正奖励 scale。

语法检查已通过：

```bash
python3 -m py_compile \
  legged_gym/legged_gym/envs/blackW/blackW_env.py \
  legged_gym/legged_gym/envs/blackW/blackW_config.py
```

注意：当前仅完成代码层面的结构改动和语法验证，尚未完成重新训练，因此还不能得出策略性能结论。

## 6. 预期效果

预期训练表现：

1. `tracking_lin_vel_y` 和 `tracking_ang_vel` 不再需要非常大的 scale。
2. 静止策略无法通过非活跃轴获得大量正奖励。
3. yaw 指令下，轮子差速会提供更直接的动作通道。
4. yaw-only 不响应会更容易被 stuck reset 清除。
5. gait/air-time/Raibert 不再主导总 reward，减少原地踏步拿高分的可能。

预期 `play` 或部署表现：

1. x 指令主要体现为轮子滚动前进/后退。
2. yaw 指令应更容易出现左右轮差速和机身旋转。
3. y 指令仍主要依赖腿部侧向步态，但 reward 不再通过大 scale 硬压，而是通过更干净的活跃轴奖励学习。

## 7. 后续实验建议

建议先进行短训练验证 reward 项是否符合预期，而不是直接长时间训练。

重点观察 TensorBoard 中：

- `rew_tracking_lin_vel`
- `rew_tracking_lin_vel_y`
- `rew_tracking_ang_vel`
- `rew_wheel_vel_ref_tracking`
- `rew_inactive_axis_vel`
- `rew_trot`
- `rew_feet_air_time`
- `rew_raibert`
- `Train/mean_reward`
- `Policy/leg_noise_std`
- `Policy/wheel_noise_std`

建议检查以下行为：

1. x-only 指令下，`rew_tracking_lin_vel` 和 `rew_wheel_vel_ref_tracking` 应成为主要正奖励。
2. y-only 指令下，`rew_tracking_lin_vel_y` 应随横向速度提高而提高，`inactive_axis_vel` 不应异常增大。
3. yaw-only 指令下，`rew_tracking_ang_vel` 和 `rew_wheel_vel_ref_tracking` 应同时改善。
4. stand 指令下，`stand_still`、`stand_still_wheels` 应约束腿和轮子保持安静。
5. 如果 `rew_trot` 或 `rew_feet_air_time` 很高但速度跟踪不提升，说明 gait shaping 仍然偏强。

## 8. 风险与待确认点

1. 当前 `wheel_control_mode = "learned"`，策略直接输出轮速参考。虽然 reward 已经给出 x/yaw 的轮速目标，但控制输入本身仍不是 residual 模式。若训练仍不稳定，可以考虑后续切到 `"residual"`。
2. yaw 轮速差速符号依赖 `wheel_side_sign` 和 `wheel_forward_sign`。当前代码按轮名推断左右侧，并沿用已有前进方向符号。后续需要通过可视化或日志确认正 yaw 指令下转向方向正确。
3. y 方向横移仍然主要依赖腿部动作。若轮子过重导致抬腿仍困难，后续可能需要进一步增加摆动腿最低离地惩罚、接触拖拽惩罚，或调整轮子质量/惯量、关节力矩和 PD 参数。
4. 新 reward 结构减少了“白拿分”，训练早期平均 reward 可能下降。这不一定是坏现象，关键应看 `play` 中的指令响应和各轴 tracking 曲线。
5. 当前没有调整命令课程。若 y/yaw 仍然难学，建议下一步加入 y/yaw 独立 curriculum，从小横移、小角速度逐步增大。

## 9. 实验结论记录位

本节预留给后续训练结果填写。

待记录：

- 训练命令：
- 训练起始 checkpoint：
- 训练迭代数：
- 平均 reward 趋势：
- y-only play 表现：
- yaw-only play 表现：
- mixed command 表现：
- 是否出现原地不响应：
- 是否出现原地踏步拿高分：
- 是否需要进一步调 scale 或课程：
