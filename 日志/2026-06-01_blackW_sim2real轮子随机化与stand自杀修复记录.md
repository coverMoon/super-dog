# blackW sim2real 轮子随机化与 stand 自杀修复记录

日期：2026-06-01  
任务：改善 `blackW` 轮足机器人 sim2real 差异，并修复 0 指令下训练后期主动终止问题  
涉及文件：

- `legged_gym/legged_gym/envs/blackW/blackW_config.py`
- `legged_gym/legged_gym/envs/blackW/blackW_env.py`
- `legged_gym/legged_gym/envs/base/legged_robot.py`

## 1. 背景

`blackW` 在 plane 地形上的 sim2sim 基本成功，能够以正常姿态在平地行进。但 sim2real 差异明显：

```text
关节感觉偏软；
重心不稳；
指令稍大时容易失衡；
从运动到停止时惯性较大，不容易停下；
后退时重心靠后，后腿被压下去，有坐下去的趋势；
旋转不稳；
横移时抬脚抬不起来。
```

因为同一台机器人在普通四足形态下已经实机成功，且腿部电机没有改变，本轮判断重点从泛化的控制链路问题转向：

```text
加轮子后足端质量和惯量变大；
轮子电机与腿部电机不同；
轮子底层通信链路可能更不可靠；
轮子质量、惯量、半径、安装位置和 URDF 存在偏差。
```

## 2. PD 实验调整

本轮先尝试提高腿部 PD 来验证“足端变重导致腿部撑不住/抬不动”的假设。

当前工作区最终保留的是一个相对温和的 PD 档：

```python
stiffness = {
    'hip_joint': 50.0,
    'thigh_joint': 50.0,
    'calf_joint': 50.0,
    'wheel_joint': 0.0,
}

damping = {
    'hip_joint': 1.2,
    'thigh_joint': 1.2,
    'calf_joint': 1.2,
    'wheel_joint': 1.0,
}
```

讨论中曾考虑过更硬的档位：hip `60/1.5`，thigh/calf `50/1.2`，wheel damping `2.0`。后续当前配置已收回到 `50/1.2` 与 wheel damping `1.0`。

## 3. 轮子专用域随机化

原有 domain randomization 大多是整机级别：

- `kp_range / kd_range` 是每个 env 一个标量，所有关节共用；
- `motor_strength_range` 原先配置存在，但在 `blackW` torque 里未实际乘上；
- `link_mass_range / inertia_range` 会覆盖所有 link，但不专门强调轮子；
- `delay` 是整条 action 共用延迟，不模拟轮子链路更差；
- `friction_range` 是所有 shape 共用，不区分轮地接触。

本轮新增 wheel-specific 随机化配置：

```python
randomize_wheel_delay = True
wheel_lag_timesteps = 6

randomize_wheel_motor = True
wheel_motor_strength_range = [0.6, 1.2]
wheel_vel_ref_scale_range = [0.7, 1.2]

randomize_wheel_mass = True
wheel_mass_scale_range = [1.0, 1.8]
wheel_inertia_scale_range = [1.0, 2.0]

randomize_wheel_geometry = True
wheel_radius_scale_range = [0.95, 1.05]
wheel_base_half_width_scale_range = [0.95, 1.05]
```

### 3.1 轮子独立延迟

`BlackWEnv.step()` 覆盖了 `BlackEnv.step()`：

1. 先按普通 `lag_buffer` 得到腿部延迟 action；
2. 再按 `wheel_lag_buffer` 得到轮子延迟 action；
3. 只替换 `wheel_indices` 对应的 action 维度。

这样可以模拟：腿部通信可靠，但轮子链路更慢、更不稳定。

### 3.2 轮子 motor / velocity scale

新增运行时 buffer：

```python
wheel_motor_strength_factors
wheel_vel_ref_scales
```

`wheel_vel_ref_scales` 作用于 learned wheel velocity reference 或 residual wheel velocity reference；`wheel_motor_strength_factors` 只作用于轮子 torque。

同时修复 `blackW` 中原有 `motor_strength_factors` 没有实际乘到 torque 的问题：

```python
torques = torques * self.motor_strength_factors
torques[:, self.wheel_indices] *= self.wheel_motor_strength_factors
```

注意：轮子的有效 torque scale 是全局 `motor_strength_factors` 和 wheel-specific `wheel_motor_strength_factors` 的乘积。

### 3.3 轮子质量和惯量

为了按刚体名字找到 wheel body，`LeggedRobot._create_envs()` 中保存了：

```python
self.body_names = body_names
```

`BlackWEnv._process_rigid_body_props()` 在父类随机化之后，再对名字包含 `wheel` 的刚体叠加 wheel-specific mass / inertia scale。

因为该随机化叠加在原有 `link_mass_range` 和 `inertia_range` 之后，轮子质量和惯量的实际随机范围会比普通 link 更宽。这符合“轮子足端更重/安装不准”的 sim2real 假设，但后续如果训练过难，可以优先收窄这两项。

### 3.4 轮子几何估计误差

`_target_wheel_velocities()` 不直接改变物理几何，而是在控制目标计算中使用随机 scale：

```python
wheel_radius_scale
wheel_base_half_width_scale
```

用于模拟轮径误差、轮距估计误差，以及 yaw 差速目标和真实运动学之间的偏差。

## 4. 0 指令下主动终止问题

后续训练观察到一个新问题：训练一段时间后，指令为 0 时机器人会主动“自杀”。相关日志现象包括：

- `mean_episode_length` 下降；
- `rew_stand_still` 相关项变差；
- `rew_stand_still_wheels` 相关项变差。

分析后判断：这不是 `stand_still` 和 `stand_still_wheels` 之间直接冲突，而是 0 指令状态成为了负收益状态。

0 指令时：

```text
tracking_lin_vel / tracking_lin_vel_y / tracking_ang_vel 被 command activity 门控，基本不给正奖励；
wheel_vel_ref_tracking 不生效；
feet_air_time / foot_clearance / raibert 等运动相关 shaping 基本不提供收入；
stand_still、stand_still_wheels、inactive_axis_vel、orientation、base_height 等惩罚仍会持续累计。
```

另外，`termination` scale 会在 `_prepare_reward_function()` 中乘以 `dt`。当前 `dt = 0.02`，所以：

```text
termination = -1000.0
实际单次终止惩罚约为 -20
```

如果 0 指令下继续存活的未来累计负回报大于一次终止惩罚，且 reset 后大概率采样到运动指令并获得 tracking 正奖励，策略就会学到通过主动终止逃离 stand command。

## 5. 保留的修复方案

### 5.1 stand_alive 正奖励

新增 stand 状态正奖励：

```python
stand_alive = 5.0
```

实现：

```python
def _reward_stand_alive(self):
    return self._stand_command_mask().float() * (~self.reset_buf).float()
```

含义：0 指令且没有触发 reset 时，提供明确正收益。这样 stand 不再只是“少扣分”，而是一个有正向目标的任务状态。

### 5.2 失败 reset 保留 command

新增配置：

```python
preserve_failed_reset_commands = True
```

在 `BlackWEnv.reset_idx()` 中：

```python
previous_commands = self.commands[env_ids].clone()
failed_reset_mask = ~self.time_out_buf[env_ids]

super().reset_idx(env_ids)

if torch.any(failed_reset_mask):
    failed_env_ids = env_ids[failed_reset_mask]
    self.commands[failed_env_ids] = previous_commands[failed_reset_mask]
```

含义：

```text
正常 timeout reset：重新采样 command；
碰撞、摔倒、stuck 等失败 reset：保留 reset 前 command。
```

这样如果机器人在 0 指令下主动终止，reset 后仍然是 0 指令，不能通过死亡换取运动指令和 tracking 正奖励。

这是比单纯增大 `termination` 更结构性的修复。

### 5.3 stand_still_wheels 保持直接写法，并降低权重

曾短暂尝试给 `_reward_stand_still_wheels()` 增加 `stand_wheel_grace_s`，用于在 command 切到 0 后给轮子惯性刹停一个 grace/ramp。

该方案已按当前要求回退。当前保留写法为：

```python
def _reward_stand_still_wheels(self):
    is_still = self._stand_command_mask()
    wheel_vel_error = torch.sum(torch.abs(self.dof_vel[:, self.wheel_indices]), dim=1)
    return wheel_vel_error * is_still
```

配置中不再保留 `stand_wheel_grace_s`。

最终缓解 0 指令自杀问题的关键调整，是将 `stand_still_wheels` 权重适当降低：

```python
stand_still_wheels = -1.0 -> -0.5
```

判断：0 指令下轮子停止惩罚原本过强，尤其在轮子存在惯性、延迟和随机化时，会让 stand command 的长期回报偏负。降低该项后，策略不再倾向于通过失败 reset 逃离 0 指令。

## 6. 当前工作区其他训练配置

当前工作区还包含一些训练/实验参数调整，需要和本轮结果一起看：

```python
stand_command_prob = 0.15
x_command_prob = 0.0
y_command_prob = 0.0
yaw_command_prob = 0.0
mixed_command_prob = 0.85

tracking_lin_vel = 8.0
tracking_lin_vel_y = 8.0
tracking_ang_vel = 8.0
progress = 2.0
stand_alive = 5.0

stand_still = -1.5
stand_still_wheels = -0.5

action_rate = -0.1
wheel_action_rate = -0.1
wheel_smoothness = -0.02

terrain_adaptive.enabled = False
entropy_coef = 0.003
max_iterations = 1000
```

这些调整会影响训练曲线，复盘时不应只看 wheel randomization 或 stand 修复本身。

## 7. 验证

语法检查通过：

```bash
python -m py_compile   legged_gym/legged_gym/envs/blackW/blackW_config.py   legged_gym/legged_gym/envs/blackW/blackW_env.py
```

## 8. 后续观察重点

TensorBoard 重点看：

- `Train/mean_episode_length`
- `Episode/rew_stand_alive`
- `Episode/rew_stand_still`
- `Episode/rew_stand_still_wheels`
- `Episode/rew_termination`
- `Episode/rew_tracking_lin_vel`
- `Episode/rew_tracking_lin_vel_y`
- `Episode/rew_tracking_ang_vel`
- `Episode/rew_wheel_vel_ref_tracking`

行为观察重点：

1. 0 指令下是否还会主动摔倒/碰撞 reset。
2. 失败 reset 后是否确实保留原 command，尤其是 stand command。
3. 轮子随机化增强后，策略是否仍能在 plane 上稳定学习。
4. sim2real 中是否改善“腿软、后退坐下、横移抬脚不足、停止惯性大”等现象。

## 9. 风险与调参建议

1. wheel mass/inertia 随机化是叠加在全局 link mass/inertia 随机化上的，若训练明显变难，可先收窄：

```python
wheel_mass_scale_range = [1.0, 1.4]
wheel_inertia_scale_range = [1.0, 1.6]
```

2. wheel delay 当前最大到 6 个 policy step，如果学习出现明显滞后或停止困难，可尝试：

```python
wheel_lag_timesteps = 4
```

3. 如果 0 指令仍出现自杀，优先检查 `stand_still_wheels` 是否过强，其次检查 `preserve_failed_reset_commands` 是否实际生效，而不是继续增大 `termination`。

4. 如果 stand 变得过于保守、影响从 stand 到运动的启动，可适当降低：

```python
stand_alive = 2.0 ~ 4.0
```

## 10. yaw 主导门控补充

同日后续排查 `May31_23-04-04_` 时发现，`Episode/rew_yaw_contact_hip_deviation` 全程为 `0.0`。原因是旧 `_yaw_activity()` 要求 yaw 活跃且 xy 线速度指令范数小于 `yaw_lin_cmd_threshold`，相当于只在纯 yaw / 近似纯 yaw 下打开；而当时训练配置以 mixed command 为主，mixed command 又必定带非零 x/y，所以该 reward 被门控为 0。

本次将 `_yaw_activity()` 从纯 yaw 门控：

```text
yaw_activity * lin_quiet
```

改为 yaw 分量 / yaw 主导门控：

```text
yaw_activity * abs(yaw_cmd) / (abs(yaw_cmd) + yaw_dominance_lin_vel_scale * norm(cmd_xy))
```

新增配置：

```python
yaw_dominance_lin_vel_scale = 1.0
```

含义：只要 yaw 指令超过 deadzone，该项就可能非零；yaw 相对 xy 越主导，权重越接近 1；xy 线速度越大且 yaw 越小，权重越弱；纯 yaw 时仍接近满权重。

影响范围：`_yaw_activity()` 当时同时影响 `yaw_contact_hip_deviation` 和 `wheel_vel_ref_tracking` 中的 yaw 下轮速追踪衰减。因此 mixed yaw 场景中髋关节接触变形惩罚会开始生效，同时轮速 tracking 衰减也会按 yaw 主导程度部分生效。

观察重点：

- `Episode/rew_yaw_contact_hip_deviation` 是否从 0 变为负值；
- `Episode/rew_wheel_vel_ref_tracking` 是否明显下降；
- `Episode/rew_tracking_ang_vel` 是否保持；
- play 中 mixed yaw / 大 yaw 指令下髋关节外展内收变形是否减轻。

如果 mixed yaw 下轮速 tracking 下降明显，应减弱 mixed 中的 yaw 作用：

```python
yaw_dominance_lin_vel_scale = 1.5 ~ 2.0
```

如果希望 yaw hip penalty 更积极介入 mixed yaw，可降低该值：

```python
yaw_dominance_lin_vel_scale = 0.5
```

## 11. yaw 去差速轮速先验补充

同日后续训练 `Jun01_10-48-56_` 500 轮后观察到：

```text
1. 零指令时仍会缓慢前滚，轮子无法完全停下，后半身轻微下坠；
2. yaw 指令下旋转不协调，前后轮有时会撞在一起。
```

其中 yaw 问题的核心判断是：`blackW` 的四个轮子是长方形布局的非转向轮，纯 yaw 下如果继续使用差速小车式轮速先验，会鼓励四轮贴地硬拧，容易产生侧向约束冲突、髋关节变形和前后轮互相靠近/碰撞。

本次让 `_target_wheel_velocities()` 支持是否包含 yaw：

```python
def _target_wheel_velocities(self, include_yaw=True):
```

默认仍保留原行为，避免影响 residual wheel control 等其它调用路径。

`_reward_wheel_vel_ref_tracking()` 改为使用不含 yaw 的 wheel target：

```python
target = self._target_wheel_velocities(include_yaw=False)
move_cmd = self._command_activity(self.commands[:, 0])
return tracking * move_cmd
```

含义：

- wheel velocity tracking 只服务 x 方向滚动；
- 纯 yaw 指令下不再奖励差速轮速；
- yaw 旋转需要通过 `tracking_ang_vel`、`raibert`、`trot`、`foot_clearance`、`feet_air_time` 等步态相关项来完成。

同时删除已无代码引用的：

```python
yaw_wheel_tracking_scale = 0.2
```

后续观察重点：

- `Episode/rew_wheel_vel_ref_tracking`：纯 yaw 占比高时应下降，因为 yaw 不再贡献该奖励；
- `Episode/rew_tracking_ang_vel`：短期可能下降，表示策略需要重新学习用步态转向；
- `Episode/rew_foot_clearance`、`Episode/rew_feet_air_time`、`Episode/rew_trot`、`Episode/rew_raibert`：观察是否开始接管 yaw；
- `Episode/rew_yaw_contact_hip_deviation`：如果仍明显变差，说明还需要更直接的足端/轮距约束；
- play 中纯 yaw 时是否减少四轮贴地硬拧和前后轮碰撞。

若 yaw tracking 掉得明显但姿态更自然，可以继续强化步态侧：提高 yaw 下 foot_clearance / feet_air_time / trot / raibert 的作用，降低 yaw curriculum 速度或 max yaw，或增加同侧前后轮最小距离、leg crossing、yaw foothold 等几何约束。

若 yaw tracking 掉得明显且姿态也没有改善，说明还需要实施第二步：把 y/yaw 统一成 lateral-yaw gait mode，而不是只切 wheel prior。

## 12. Raibert / foothold 名义落足点轮子几何补偿

同日继续检查 `raibert` 与 `foothold` 这类依赖名义落足点的 reward。`blackW` 加轮子后，`foot_name = "foot"` 实际匹配的是 `FL_foot/FR_foot/RL_foot/RR_foot` 这些轮子 link；而原先名义落足点仍沿用普通足端：

```python
nominal_front_x = 0.21
nominal_rear_x = -0.21
nominal_y = 0.155
```

根据 URDF 默认关节角估算，轮子圆柱中心默认 XY 近似为：

```text
FL: ( 0.2055,  0.1834)
FR: ( 0.2055, -0.1834)
RL: (-0.2197,  0.1834)
RR: (-0.2197, -0.1834)
```

因此本次先采用硬编码方式，将 `BlackWCfg.rewards.raibert` 中的名义落足点改为轮子几何中心近似值：

```python
nominal_front_x = 0.2055
nominal_rear_x = -0.2197
nominal_y = 0.1834
```

影响范围：

- `_reward_foothold()` 的 nominal reference 外移到轮子中心；
- `_reward_raibert()` 的 lateral/yaw target 也围绕轮子中心生成；
- yaw 时 `yaw_basis = (-nominal_y, nominal_x)` 会使用更宽的轮距，和真实轮子接触几何更一致。

风险：这只是先把数值补对的硬编码方案，并没有显式区分 link origin、COM、collision center 或轮子接触点。如果后续仍发现 yaw 下前后轮靠近、脚端几何不稳定，可以再把 wheel foothold point 单独算成 `foot_pos + local_offset`，而不是只改 nominal。


## 13. tracking min_ref 除零 NaN 修复

同日训练时报错：actor 输出 `Normal(loc=...)` 中 `loc` 全部为 NaN。排查当前 diff 后发现：

```python
relative_tracking_min_lin_cmd = 0.0
relative_tracking_min_yaw_cmd = 0.0
```

而 `_axis_tracking_progress_reward()` 中有：

```python
ref = torch.clamp(torch.abs(command), min=min_ref)
rel_error = (command - actual) / ref
progress = torch.clamp(signed_progress / ref, min=0.0, max=1.0)
```

当某个轴 command 为 0 且 `min_ref = 0` 时，`ref = 0`，会产生除零 NaN/Inf。即使 command activity 为 0，后续 `0 * NaN` 仍会得到 NaN，最终污染 reward/advantage 并把 actor 网络更新成 NaN。

本次修复：

```python
relative_tracking_min_lin_cmd = 0.2
relative_tracking_min_yaw_cmd = 0.3
```

并在代码里加防御性下限：

```python
ref = torch.clamp(torch.abs(command), min=max(min_ref, 1e-6))
```

结论：这个 NaN 与“静止时约 0.2m/s 前滑”不是同一个问题。训练采样逻辑里 stand command 会先清零，`min_nonzero_lin_cmd = 0.2` 只会用于 x/y/yaw/mixed 非零指令，不会覆盖 stand command。

## 14. tracking reward inactive 轴跳过除法

同日进一步调整 tracking reward：希望 `relative_tracking_min_lin_cmd` 和 `relative_tracking_min_yaw_cmd` 可以设为 `0.0` 或尽量接近 0，不再用 `0.2/0.3` 作为低速指令的最小归一化参考。

为避免零指令轴除零，`_axis_tracking_progress_reward()` 改为先根据 command activity 过滤：

```python
activity = self._command_activity(command)
active = activity > 0.0
if not torch.any(active).item():
    return torch.zeros_like(command)
```

后续 `ref`、`rel_error`、`progress` 只在 `active` 样本上计算：

```python
ref = torch.clamp(torch.abs(command[active]), min=max(min_ref, 1e-6))
```

这样 activity 为 0 的环境直接返回 0，不再参与除法，也不会出现 `0 * NaN`。

配置改为：

```python
relative_tracking_min_lin_cmd = 0.0
relative_tracking_min_yaw_cmd = 0.0
```

含义：非零且超过 deadzone 的低速指令会按真实指令幅值做相对误差归一化；零指令轴不进入 tracking/progress 计算，由 `inactive_axis_vel`、stand 相关 reward 等负责约束。
