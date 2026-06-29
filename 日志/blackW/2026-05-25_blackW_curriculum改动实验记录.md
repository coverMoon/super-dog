# blackW command curriculum 改动实验记录

日期：2026-05-25  
任务：`blackW` 轮足机器人 x/y/yaw 速度跟踪课程学习  
涉及文件：

- `legged_gym/legged_gym/envs/blackW/blackW_config.py`
- `legged_gym/legged_gym/envs/blackW/blackW_env.py`
- `legged_gym/legged_gym/envs/base/legged_robot.py`

## 1. 背景

在 reward 结构改动后，连续训练了多轮 500 iteration：

| 原目录 | 主要设置 | 现象 |
| --- | --- | --- |
| `May25_14-48-50_` | 4096 env，旧全范围 y/yaw | x tracking 上升，y/yaw 和 Raibert/Trot 下降 |
| `May25_15-21-47_` | 8192 env，旧全范围 y/yaw | 与 4096 形态一致，说明问题不在 env 数 |
| `May25_17-35-09_` | y/yaw 初始小范围，tracking 权重升高 | y/yaw 明显改善，但 command curriculum 没推进 |
| `May25_19-25-15_` | 阈值/EMA/buffer 调整 | 使用了更激进的课程参数，仍需要进一步观察 |

训练中观察到：

1. `play` 里 x 速度跟踪基本准确，但 TensorBoard 里 `rew_tracking_lin_vel` 看起来并不接近权重上限。
2. y 轴已经能走几步，但 `rew_tracking_lin_vel_y` 仍看起来偏低。
3. x/y/yaw 的 `max_command_*` 没有升级，课程推进为 0。

分析后确认：TensorBoard 的 reward 是所有 command 类型混合平均。当前命令采样约为：

```python
stand_command_prob = 0.1
x_command_prob = 0.3
y_command_prob = 0.3
yaw_command_prob = 0.3
mixed_command_prob = 0.0
```

因此即使 x-only 在 `play` 中满分，训练日志中的 x tracking 理论上限也约为 `tracking_lin_vel_scale * 0.3`，不是完整 scale。

更关键的问题是旧 command curriculum 用整段 episode 统计，而 episode 长度为 20s，命令重采样时间为 10s：

```python
episode_length_s = 20
resampling_time = 10.0
```

一个 episode 内可能包含两个不同 command 段。旧逻辑用整段 episode 的 tracking 累积去匹配 reset 时刻的当前 command，导致前后两个命令段互相稀释，课程判据不可靠。

## 2. 改动目标

本次改动目标：

```text
课程学习不再按完整 episode 统计，而是按 command segment 统计。
每次 command resample 前，先结算上一段 command 的表现。
x/y/yaw 分别维护自己的 segment buffer、EMA、pass streak 和升级状态。
```

这样 `resampling_time = 10s` 时，课程判断对应的是这 10s 内的真实指令段，不再被 episode 另一半命令污染。

## 3. 配置改动

### 3.1 y/yaw 初始课程范围

保留 x 初始范围：

```python
lin_vel_x = [-1.0, 1.0]
```

y/yaw 从更容易学习的小范围开始：

```python
lin_vel_y = [-0.3, 0.3]
ang_vel_yaw = [-0.8, 0.8]
```

目标最大范围：

```python
max_curriculum_y = 1.0
max_curriculum_yaw = 3.14
```

扩展步长：

```python
y_curriculum_step = 0.1
yaw_curriculum_step = 0.2
```

### 3.2 当前课程阈值与 EMA

当前工作区配置：

```python
curriculum_threshold = 0.6
curriculum_ema_alpha = 0.5
curriculum_required_passes = 2
curriculum_buffer_min = 128
y_curriculum_threshold = 0.45
yaw_curriculum_threshold = 0.35
```

说明：

- `curriculum_ema_alpha` 是三轴共用，计算形式为 `ema = (1 - alpha) * old + alpha * current`。
- `alpha = 0.5` 表示当前 segment batch 和历史 EMA 各占一半，比原来的 `0.2` 响应更快。
- x/y/yaw 阈值分开设置，但 EMA 平滑系数共用。

### 3.3 reward / PPO 配置调整

当前工作区还包含以下训练参数调整：

```python
tracking_lin_vel = 5.0
tracking_lin_vel_y = 5.0
tracking_ang_vel = 5.0
wheel_vel_ref_tracking = 2.0
trot = 2.0
raibert = 2.0
num_envs = 8192
entropy_coef = 0.003
learning_rate_max = 5e-3
```

这些调整来自后续训练尝试，用于增强主任务 reward 和保持探索。

## 4. 代码实现

### 4.1 segment buffer 初始化

在 `BlackWEnv._init_blackW_command_curriculum()` 中新增：

```python
cmd_curr_segment_len
cmd_curr_segment_sums["tracking_lin_vel"]
cmd_curr_segment_sums["tracking_lin_vel_y"]
cmd_curr_segment_sums["tracking_ang_vel"]
```

同时为 x/y/yaw 各自维护：

- low/high EMA
- pass streak
- command buffer
- ratio buffer
- sample count
- progressed 标记
- threshold 日志值

### 4.2 command resample 前结算 segment

在 `BlackWEnv._post_physics_step_callback()` 中：

```python
env_ids = (episode_length_buf % resample_interval == 0).nonzero(...)
_finalize_command_curriculum_segments(env_ids)
super()._post_physics_step_callback()
```

也就是说，在父类真正 `_resample_commands(env_ids)` 之前，先用旧 command 结算当前 segment。

### 4.3 reset 前结算未满 segment

在 `BlackWEnv.reset_idx(env_ids)` 中：

```python
_finalize_command_curriculum_segments(env_ids)
super().reset_idx(env_ids)
```

这样提前摔倒、卡死或 timeout 的环境也会把当前未满 10s 的 command 段纳入统计。

### 4.4 每步累加 segment reward

覆盖 `BlackWEnv.compute_reward()`：

1. 记录调用父类 reward 前的 episode sums。
2. 调用 `super().compute_reward()`。
3. 用 episode sums 的差值取出本 step 三个 tracking reward。
4. 累加到当前 command segment。
5. `cmd_curr_segment_len += 1`。

这样 segment ratio 与训练中实际 reward 完全同源。

### 4.5 三轴统一课程更新

`BlackWEnv.update_command_curriculum()` 不再调用基类 episode-level x curriculum，而是统一调用 `_update_axis_command_curriculum()`：

- x: `lin_vel_x` + `tracking_lin_vel`
- y: `lin_vel_y` + `tracking_lin_vel_y`
- yaw: `ang_vel_yaw` + `tracking_ang_vel`

每个 segment 只有在对应轴命令大于最小非零命令时才进入该轴 buffer：

```python
abs(command_axis) >= min_nonzero_cmd
```

这避免 x 被 y/yaw/stand segment 稀释，也避免 y/yaw 被无关命令污染。

### 4.6 日志字段

在 `LeggedRobot.reset_idx()` 的 `extras["episode"]` 中保留原 x 字段，并新增 y/yaw 字段。

x：

- `Episode/max_command_x`
- `Episode/cmd_curr_low_ratio_ema`
- `Episode/cmd_curr_high_ratio_ema`
- `Episode/cmd_curr_ratio_ema`
- `Episode/cmd_curr_pass_streak`
- `Episode/cmd_curr_sample_count`
- `Episode/cmd_curr_low_count`
- `Episode/cmd_curr_progressed`
- `Episode/cmd_curr_threshold_ratio`

y：

- `Episode/max_command_y`
- `Episode/cmd_curr_y_low_ratio_ema`
- `Episode/cmd_curr_y_high_ratio_ema`
- `Episode/cmd_curr_y_ratio_ema`
- `Episode/cmd_curr_y_pass_streak`
- `Episode/cmd_curr_y_sample_count`
- `Episode/cmd_curr_y_low_count`
- `Episode/cmd_curr_y_progressed`
- `Episode/cmd_curr_y_threshold_ratio`

yaw：

- `Episode/max_command_yaw`
- `Episode/cmd_curr_yaw_low_ratio_ema`
- `Episode/cmd_curr_yaw_high_ratio_ema`
- `Episode/cmd_curr_yaw_ratio_ema`
- `Episode/cmd_curr_yaw_pass_streak`
- `Episode/cmd_curr_yaw_sample_count`
- `Episode/cmd_curr_yaw_low_count`
- `Episode/cmd_curr_yaw_progressed`
- `Episode/cmd_curr_yaw_threshold_ratio`

## 5. 已分析的训练现象

### 5.1 `May25_14-48-50_` 与 `May25_15-21-47_`

两轮配置除了 `num_envs = 4096/8192` 外基本一致。结果同形：

- `tracking_lin_vel` 上升。
- `tracking_lin_vel_y`、`tracking_ang_vel`、`raibert`、`trot` 下降。
- episode length 接近满长，termination 减少。

判断：策略学到了更稳定、更少动、更不摔的局部最优，而不是学到全向跟踪。问题不在 env 数。

### 5.2 `May25_17-35-09_`

该轮启用了 y/yaw 小范围课程，并提高 tracking 权重：

```python
tracking_lin_vel = 5.0
tracking_lin_vel_y = 5.0
tracking_ang_vel = 5.0
```

最后 20 个点约为：

- `Episode/rew_tracking_lin_vel`：1.17
- `Episode/rew_tracking_lin_vel_y`：0.86
- `Episode/rew_tracking_ang_vel`：0.65

由于 x/y/yaw 各自采样概率约为 0.3，单轴 tracking 的混合日志理论上限约为 `5.0 * 0.3 = 1.5`。因此 x 的日志值并不低，约等于理论上限的 78%。

y/yaw 相对旧轮明显改善，但 `max_command_y` 和 `max_command_yaw` 未推进。原因是旧 curriculum 统计口径仍然是 episode-level，且阈值偏严。

### 5.3 `May25_19-25-15_`

该轮使用了更快 EMA 和更低 buffer：

```python
curriculum_ema_alpha = 0.5
curriculum_buffer_min = 128
curriculum_threshold = 0.6
yaw_curriculum_threshold = 0.35
```

该轮仍属于 segment curriculum 改动前的训练结果，不能代表当前最新代码效果。后续应以 segment curriculum 版本重新训练评估。

## 6. 验证

代码语法检查通过：

```bash
python -m py_compile   legged_gym/legged_gym/envs/blackW/blackW_config.py   legged_gym/legged_gym/envs/blackW/blackW_env.py   legged_gym/legged_gym/envs/base/legged_robot.py
```

注意：当前 shell 直接 import 任务配置会因缺少 `isaacgym.torch_utils` 失败，这是环境依赖问题，不是语法问题。

## 7. 下一轮训练观察重点

重点看课程是否真正推进：

- `Episode/max_command_x`
- `Episode/max_command_y`
- `Episode/max_command_yaw`
- `Episode/cmd_curr_ratio_ema`
- `Episode/cmd_curr_y_ratio_ema`
- `Episode/cmd_curr_yaw_ratio_ema`
- `Episode/cmd_curr_progressed`
- `Episode/cmd_curr_y_progressed`
- `Episode/cmd_curr_yaw_progressed`

同时继续看任务行为：

- x-only play：是否稳定跟踪目标速度。
- y-only play：是否能持续横移，而不是只走一两步。
- yaw-only play：是否能持续朝目标方向旋转。
- stand：腿和轮子是否安静。

## 8. 风险与注意事项

1. 当前 x/y/yaw 共用 `curriculum_ema_alpha`，如果后续发现 y/yaw 需要更快响应，可以拆成 `y_curriculum_ema_alpha` 和 `yaw_curriculum_ema_alpha`。
2. `cmd_curr_segment_len` 统计的是训练 rollout 下的 stochastic policy 表现；`play` 通常更接近 deterministic policy，所以 play 表现可能优于训练日志。
3. segment curriculum 使用 reward scale 归一化，三轴 tracking scale 同时调高不会直接改变 curriculum ratio，但会影响 PPO 优化压力。
4. `config.json` 中保存的 `log_dir` 是训练时旧目录快照，如果后续重命名训练目录，不应反向修改历史快照。
