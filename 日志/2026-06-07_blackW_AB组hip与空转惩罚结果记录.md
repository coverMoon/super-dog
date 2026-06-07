# 2026-06-07 blackW A/B 组 hip 与空转惩罚结果记录

## 1. Jun06_18-18-57_ 与 Jun06_18-32-14_ 对比

实验背景：两组都从 Jun05_16-19-29_ 续训，并启用 hip_default 绝对值误差、wheel_obstacle_spin、stand wheel 约束和 wheel-specific friction randomization。

配置确认：

- Jun06_18-18-57_ 为 A 组：hip_default = -0.25，wheel_obstacle_spin = -0.03。
- Jun06_18-32-14_ 为 B 组：hip_default = -0.35，wheel_obstacle_spin = -0.06。
- 两组均为 smoothness = -0.01，action_rate = -0.06，torques = -6.2e-4，wheel_obstacle_lift = 1.2。
- 两组均启用 randomize_wheel_friction，wheel_friction_scale_range = [0.4, 1.0]，全局 friction_range = [0.25, 1.25]。

关键结果：

- A 组后段明显失稳：25000 step 时 mean_reward 约 -112.6，terrain_level 约 3.87，tracking_lin_vel_x 约 0.77，tracking_ang_vel 约 0.41，value_loss 约 792，mean_noise_std 升到约 0.75。
- A 组从 22000 step 左右开始出现明显退化，stand_wheel_action、torques、smoothness、base_height 逐步恶化，terrain_level 后段从约 5.5 掉到 3.9。
- B 组整体稳定：25000 step 时 mean_reward 约 39.0，terrain_level 约 5.52，tracking_lin_vel_x 约 1.16，tracking_lin_vel_y 约 1.24，tracking_ang_vel 约 0.61，value_loss 末尾约 0.18。
- B 组后段 tracking_ang_vel 和 mean_reward 相比父 run 有一定下降，mean_noise_std 升到约 0.55，但没有出现 A 组那种 value loss 和 terrain_level 崩溃。
- B 组 wheel_obstacle_spin tail100 均值约 -0.048，A 组 tail100 约 -0.109；A 组虽 scale 更小但后段行为变差导致空转惩罚更大。
- B 组 rew_hip_default tail100 约 -0.143，A 组约 -0.188；A 组 hip 偏差后段更大，说明较弱 hip_default 没有带来更自由有效的动作，反而随失稳变差。

判断：

- 这轮结果不支持 A 组的低风险假设；A 组约束偏弱，训练后段漂到高力矩、高 smoothness、高 stand wheel action 的坏模式。
- B 组虽然 hip_default 与 wheel_obstacle_spin 更强，但训练稳定性显著更好，越障/terrain_level 保持接近父 run。
- wheel_obstacle_spin 数值大小不能只看 scale；A 组后段惩罚更大主要是策略失稳和空转行为增多，而不是 anti-spin 更强。

后续建议：

- 优先保留 B 组作为当前候选：hip_default = -0.35，wheel_obstacle_spin = -0.06。
- 若担心 B 组 yaw/tracking_ang_vel 下降，可在 B 组基础上微调 hip_default 到 -0.30，而不是回到 A 组。
- 暂不继续降低 wheel_obstacle_spin；若实机高墙仍空转，可优先看 B 组部署表现，再决定是否调整 gate。



## 2. orientation L1 与 pitch 地形自适应改动

背景：B/D 组实机测试显示，姿态不稳是当前训练线的共性问题，表现为静止时向左后方或右后方偏斜、yaw 指令转向时前腿受摩擦卡住后姿态不易恢复，以及低速 y 指令下身体先侧倾而不是先抬腿。Jun02_22-30-50_ 姿态表现较好但地形较简单，说明后续复杂地形和越障训练可能让策略学会了用身体倾斜换 tracking 或通过率。

本次改动：

- 将 blackW 的 orientation reward 从 projected_gravity[:2] 平方和改为 L1 形式，增强小角度倾斜时的恢复信号。
- orientation 惩罚拆成 roll 与 pitch：roll 始终保持完整惩罚，pitch 根据机身下方高度采样的 std 做快速衰减。
- 只为 orientation 增加局部地形自适应参数，不启用完整 terrain_adaptive 框架，避免牵连 smoothness、action_rate、torques 等其他 reward。
- 配置设为 orientation = -3.0，roll_orientation = -0.0；新增 orientation_terrain_sigma = 0.004，orientation_terrain_min_scale = 0.10，orientation_terrain_max_scale = 1.0，orientation_terrain_variability_clip = 0.30。

预期：平地和低速指令下更强约束身体姿态，减少静止偏斜、yaw 卡腿后的姿态残留和 y 指令先侧倾的问题；复杂地形或高墙附近 pitch 惩罚快速放松，尽量保留越障所需的前后俯仰自由度。

观察重点：rew_orientation 数值会因 L1 与 scale 加大而不可直接和旧日志横比；重点看 terrain_level、wheel_obstacle_lift、tracking_lin_vel_y、tracking_ang_vel 是否保持，以及实机低速 y/yaw 下是否能更主动回正。


## 3. orientation 衰减参数更新与髋默认位置对照计划

在上一版 orientation L1 的基础上，又补充调整了若干训练参数：

- orientation_terrain_sigma 从 0.004 调整为 0.0015，使 pitch 惩罚随地形高度 std 衰减更快。
- orientation_terrain_min_scale 从 0.10 调整为 0.05，使明显台阶/高墙附近的 pitch 姿态惩罚可以放得更开。
- entropy_coef 从 0.001 调整为 0.003，避免 noise_std 在短续训中过快降得过低，同时仍低于此前 B 组长训时的 0.005。
- sym_coef 从 0.6 调整为 0.8，加强左右对称约束，希望减少实机静止或低速指令下向某一后侧偏斜的问题。
- max_iterations 调整为 1000，用于从当前候选模型做短续训对照。

当前准备在这套 orientation L1 + 快速 pitch 地形衰减配置基础上，对髋关节默认位置做 A/B 对照：

- A 组：不改默认髋关节位置，四个 hip 数值仍为 0.0。当前代码对应此组；后腿 hip 的 -0.0/0.0 只影响显示符号，不改变数值。
- B 组：四个 hip 统一向外放 0.05 rad，按现有四足符号约定建议设置为 FL/RL = +0.05，FR/RR = -0.05。

实验目的：验证更宽的默认支撑宽度能否减少静止偏斜、低速 y 指令先侧倾、yaw 卡腿后姿态不易恢复等问题，同时观察是否影响 tracking_lin_vel_y、tracking_ang_vel、terrain_level 和 wheel_obstacle_lift。

注意：由于策略 action 是相对 default_dof_pos 的残差，B 组会改变同一 action 对应的实际目标关节角；建议与 A 组只差默认 hip 位置，其他参数保持一致，便于判断站姿外展本身的影响。
