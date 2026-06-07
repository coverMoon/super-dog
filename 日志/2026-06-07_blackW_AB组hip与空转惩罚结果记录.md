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


## 4. Jun07_17-44-57_ 与 Jun07_17-46-13_ 结果及实机反馈

实验配置：两组均使用 orientation L1、pitch 地形自适应快速衰减、entropy_coef = 0.003、sym_coef = 0.8、max_iterations = 1000。Jun07_17-44-57_ 默认 hip 仍为 0.0；Jun07_17-46-13_ 的 hip 默认位置为 FL/RR = +0.05、FR/RL = -0.05。后者并非左侧统一为正、右侧统一为负，需要后续确认符号是否符合真实外展定义。

训练曲线：

- Jun07_17-44-57_ 末尾 mean_reward 约 64.7，terrain_level 约 5.54，tracking_ang_vel 约 0.80，mean_noise_std 约 0.26；但 value_loss 末尾升到约 40.6，tail100 约 10.7，说明后段 critic 有明显波动。
- Jun07_17-46-13_ 末尾 mean_reward 约 59.7，terrain_level 约 5.57，tracking_ang_vel 约 0.79，mean_noise_std 约 0.30；value_loss 约 0.15，训练稳定性更好。
- 两组 rew_orientation tail100 均约 -0.194，说明 orientation L1 后姿态惩罚处在相近水平；run_still tail100 均约 -0.112，对应运动时非轮关节相对默认位置仍存在较明显的累计偏差。

实机反馈：使用 Jun07_17-44-57_ 并在部署侧临时将 hip 默认位置稍微外放后，姿态问题有一定改善但没有完全解决；髋关节外放方向有效，但受重力影响会被压得更宽，可能导致大腿与小腿连接处压力增大。持续前进或后退时仍会出现关节变形，推测 run_still 约束偏弱或运动中关节回默认形态的约束不够。

后续判断：髋默认位置外放有效，但幅度不宜继续直接增大，建议优先测试更小外放量或确保符号为真正左右外展；同时考虑增强 run_still 或增加更细分的运动中关节姿态约束，用于抑制持续 x 指令下的慢性变形。


## 5. y 指令抬轮 gap 与 run_still 增强

实机反馈：部署 Jun07_17-44-57_ 时将 hip 默认位置临时外放到约 0.1 rad 后，姿态稳定性进一步改善，说明外展增大支撑宽度方向有效；当前训练配置中的 0.05 rad 外放仍可继续使用。但实机上由于重力会把髋关节压得更宽，过大的部署外放可能让大腿与小腿连接处承受额外压力。

仍存在的问题：y 指令下姿态仍不够稳，虽然 orientation L1 已有改善；更明显的 sim2real gap 是实机轮子/腿不容易主动抬起，很多时候仍在地上蹭，而仿真里轮子离地更充分。持续 x 前进或后退时也会出现关节慢性变形，说明 run_still 约束偏弱。

本次改动：

- run_still scale 从 -0.05 提到 -0.4，用于增强运动中非轮关节回默认姿态的约束，抑制持续前进/后退时的关节变形。
- 新增 wheel_lateral_clearance reward，只在 abs(cmd_y) > 0.2 时启用；以轮子中心相对地面的高度为目标，要求 top-2 个轮子达到 wheel_radius + 0.04 m 的额外离地高度。
- wheel_lateral_clearance 使用地形高度 std 做衰减，terrain_variability_sigma = 0.0015，复杂地形/高墙附近奖励快速变弱，尽量避免和 wheel_obstacle_lift 或越障动作冲突。
- 当前 wheel_lateral_clearance scale = 0.5，tracking_sigma = 0.03，top_k = 2。

预期：横向低速运动时减少轮子贴地横蹭，鼓励策略至少抬起部分轮子完成迈步；同时通过更强 run_still 抑制持续 x 指令下的腿部形变。观察重点为 tracking_lin_vel_y、orientation、run_still、wheel_lateral_clearance、terrain_level 和 wheel_obstacle_lift。


补充：run_still 原先通过 norm(cmd_x, cmd_y) 开启，会在 y 指令下同样强压非轮关节回默认位置，与横向迈步和 wheel_lateral_clearance 存在冲突。已新增 run_still_y_threshold = 0.1，并将 _reward_run_still 门控改为仅在 abs(cmd_x) > run_still_x_threshold 且 abs(cmd_y) < run_still_y_threshold 时启用，使 run_still 主要处理持续前进/后退时的关节慢性变形。


## 6. 加强 y 抬轮与 x 运动回中约束

400 轮短训后，新增 wheel_lateral_clearance 与 run_still 门控的收益不够明显，推测一方面当前 27000 轮附近模型动作模式已有一定固化，另一方面 y 指令抬轮与持续 x 运动回默认的约束仍偏弱。后续准备一条线继续从当前模型续训，另一条线使用相同配置从 Jun02_22-30-50_ 的模型继续训练，以验证更早、姿态更干净的模型是否更容易学出横向抬轮动作。

本次参数调整：

- base_height_target 从 0.53 降到 0.52，希望略微降低重心，减轻姿态漂移。
- run_still 拆分为 run_still_x_threshold 与 run_still_y_threshold：仅在 abs(cmd_x) > 0.1 且 abs(cmd_y) < 0.1 时启用，避免与 y 指令横向迈步和 wheel_lateral_clearance 冲突。
- run_still scale 加强到 -0.5，用于更强地抑制持续前进/后退时非轮关节相对默认位置的慢性变形。
- wheel_lateral_clearance scale 加强到 1.2，command_threshold 为 0.1，目标仍为 top-2 轮子达到 wheel_radius + 0.04 m 的额外离地高度。
- wheel_obstacle_lift scale 加强到 1.5，继续鼓励障碍前轮子主动抬升。
- wheel_obstacle_spin 的 spin_threshold 从 8.0 降到 6.0，使空转惩罚更早触发。

风险：run_still = -0.5 已经较强，可能压缩持续 x 运动中的步态幅度；wheel_lateral_clearance = 1.2 可能影响 y tracking 或诱导不必要抬腿。观察重点为 tracking_lin_vel_x/y、rew_run_still、rew_wheel_lateral_clearance、rew_wheel_obstacle_lift、terrain_level、姿态稳定性和实机轮子是否仍贴地横蹭。
