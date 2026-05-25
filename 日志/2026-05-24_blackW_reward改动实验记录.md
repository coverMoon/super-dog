# blackW reward 改动实验记录

日期：2026-05-24  
任务：`blackW` 轮足机器人速度跟踪训练  
涉及文件：

- `legged_gym/legged_gym/envs/blackW/blackW_config.py`
- `legged_gym/legged_gym/envs/blackW/blackW_env.py`

## 1. 背景

训练 `blackW` 时曾出现一个典型现象：训练日志中的 reward 看起来较高，但 `play` 或部署时机器人对 y 轴横移、yaw 原地旋转指令响应较弱，甚至可能保持静止或只维持姿态。

主要原因不是简单的 reward scale 不够大，而是原 reward 结构允许策略在非活跃指令轴上“白拿分”。例如 x-only 指令下，y 和 yaw 指令为 0，机器人只要保持 y/yaw 速度接近 0，就能拿到很高的 y/yaw tracking reward。如果把 y/yaw 的 scale 调得很大，这个漏洞也会被同步放大。

## 2. 原结构问题

### 2.1 非活跃轴白拿 tracking 分

旧 tracking 形式近似为：

```text
exp(-(cmd - actual)^2 / sigma)
```

当某轴 `cmd = 0` 且 `actual = 0` 时，该轴 reward 接近满分。于是策略可以在没有真正执行横移或旋转的情况下，通过保持静止获得大量正奖励。

### 2.2 y/yaw 大权重会放大奖励漏洞

曾经使用过较大的 y/yaw tracking scale：

```python
tracking_lin_vel_y = 15.0
tracking_ang_vel = 20.0
```

这确实提高了 y/yaw 项的重要性，但同时也放大了非活跃轴静止奖励，导致训练 reward 与真实指令响应脱节。

### 2.3 yaw 指令缺少轮子差速目标

旧轮速参考主要来自 `cmd_x`，yaw-only 指令没有明确差速轮速目标。对轮足机器人而言，yaw 如果完全依赖腿部摆动和接触切换，学习难度很高。

### 2.4 yaw-only 卡死检测不足

继承自 `black` 的 stuck 检测主要针对平移指令。如果 yaw-only 指令下机器人不旋转，旧逻辑不一定及时 reset，训练中会保留较多低质量样本。

### 2.5 gait shaping 容易喧宾夺主

`feet_air_time`、`trot`、`raibert` 等项本意是辅助步态，但如果权重过大，策略可能优先学习原地踏步或接触相位，而不是完成速度跟踪。

## 3. 改动目标

本轮 reward 改动的目标是让 reward 更贴近任务本身：

```text
只有当前指令轴活跃时，才给该轴主要 tracking 正奖励。
非活跃轴不再白拿正奖励，而是需要保持安静。
x/yaw 尽量通过轮子提供直接动作通道。
y 方向主要由腿部侧向步态完成。
gait/Raibert/air-time 只做辅助 shaping，不作为主要收入来源。
```

## 4. 主要实现

### 4.1 命令活跃度门控

新增 `_command_activity(command)`：

```text
activity = clamp((abs(command) - deadzone) / (full - deadzone), 0, 1)
```

配置：

```python
command_activity_deadzone = 0.05
command_activity_full = 0.2
```

含义：小于 deadzone 的命令不激活该轴 reward，大于 full 后视为完全活跃，中间平滑过渡。

### 4.2 tracking 改为相对误差 + progress

新增 `_axis_tracking_progress_reward(command, actual, min_ref)`，三轴 tracking 都走同一个 helper：

```text
ref = max(abs(command), min_ref)
relative_error = (command - actual) / ref
tracking = exp(-relative_error^2 / relative_tracking_sigma)
progress = clamp(sign(command) * actual / ref, 0, 1)
reward = activity * (tracking_reward_weight * tracking + progress_reward_weight * progress)
```

关键参数：

```python
relative_tracking_sigma = 0.25
relative_tracking_min_lin_cmd = 0.2
relative_tracking_min_yaw_cmd = 0.3
tracking_reward_weight = 0.6
progress_reward_weight = 0.4
```

设计含义：tracking 负责精确跟踪，progress 保证训练早期只要朝正确方向动起来就有梯度。

### 4.3 新增非活跃轴速度惩罚

新增 `_reward_inactive_axis_vel`：

```text
penalty =
    (1 - active_x)   * vx^2
  + (1 - active_y)   * vy^2
  + (1 - active_yaw) * inactive_ang_vel_weight * wz^2
```

配置：

```python
inactive_axis_vel = -0.5
inactive_lin_vel_weight = 1.0
inactive_ang_vel_weight = 0.25
```

效果：x-only 时约束 y/yaw，y-only 时约束 x/yaw，yaw-only 时约束 x/y，stand 时三轴都保持安静。

### 4.4 yaw 加入轮子差速参考

`_target_wheel_velocities()` 从仅使用 x 速度改为同时使用 x 和 yaw：

```text
target = (cmd_x - cmd_yaw * wheel_side_sign * wheel_base_half_width) / wheel_radius
target = target * wheel_forward_sign
```

这样 yaw-only 指令能给左右轮差速提供明确参考。

### 4.5 wheel tracking 覆盖 x/yaw

`_reward_wheel_vel_ref_tracking()` 改为跟踪 `_target_wheel_velocities()`，并只在 x 或 yaw 指令活跃时生效。相对误差参数：

```python
wheel_tracking_relative_sigma = 0.25
wheel_tracking_min_ref = 0.5
```

### 4.6 修复 stand_still_wheels 与 yaw-only 冲突

旧逻辑主要看 `cmd_x` 是否接近 0。新逻辑要求 x/y/yaw 三轴都接近 0 时，才惩罚轮子转动：

```python
is_still = norm(cmd_xy) < 0.1 and abs(cmd_yaw) < 0.1
```

yaw-only 时轮子差速是期望行为，不再被 stand-still wheel 惩罚误伤。

### 4.7 yaw-only 卡死检测

在 `BlackWEnv.check_termination()` 中补充 yaw 卡死检测：

```text
yaw_cmd = abs(cmd_yaw) > stuck_command_threshold
yaw_progress = sign(cmd_yaw) * base_ang_vel_z
yaw_stalled = yaw_progress < stuck_yaw_vel_threshold
```

配置：

```python
stuck_yaw_vel_threshold = 0.1
```

长期不朝目标方向旋转时触发 stuck reset。

### 4.8 reward scale 回到相对均衡

改动后的基准权重：

```python
tracking_lin_vel = 2.0
tracking_lin_vel_y = 2.0
tracking_ang_vel = 2.0
wheel_vel_ref_tracking = 1.0
progress = 0.0
feet_air_time = 0.8
trot = 1.0
raibert = 1.0
```

旧的独立 `progress` 关闭，因为 progress 已经并入每个轴的 tracking-progress reward。

## 5. 验证

语法检查通过：

```bash
python3 -m py_compile   legged_gym/legged_gym/envs/blackW/blackW_env.py   legged_gym/legged_gym/envs/blackW/blackW_config.py
```

当时尚未完成重新训练，因此该记录只说明 reward 结构改动，不直接给出策略性能结论。

## 6. 后续观察重点

TensorBoard 重点看：

- `Episode/rew_tracking_lin_vel`
- `Episode/rew_tracking_lin_vel_y`
- `Episode/rew_tracking_ang_vel`
- `Episode/rew_wheel_vel_ref_tracking`
- `Episode/rew_inactive_axis_vel`
- `Episode/rew_trot`
- `Episode/rew_raibert`
- `Train/mean_reward`
- `Policy/leg_noise_std`
- `Policy/wheel_noise_std`

`play` 重点检查：

1. x-only 是否主要通过轮子稳定前后运动。
2. y-only 是否出现真实侧向步态，而不是静止拿分。
3. yaw-only 是否出现左右轮差速和机身旋转。
4. stand 是否能约束腿和轮子保持安静。

## 7. 风险与待确认

1. 当前仍使用 `wheel_control_mode = "learned"`，轮速参考通过 reward 牵引，而不是直接 residual 控制。
2. yaw 轮速方向依赖 `wheel_side_sign` 和 `wheel_forward_sign`，需要通过可视化确认正负方向。
3. y 方向仍主要依赖腿部侧向动作，如果轮子惯量和接触阻力过大，后续可能还需要调整机械参数、PD、接触/摆腿相关 shaping。
4. 新 reward 减少了白拿分，训练早期 reward 下降不一定是坏事，关键应看各轴指令响应。
