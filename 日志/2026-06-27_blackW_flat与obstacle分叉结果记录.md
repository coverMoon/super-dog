# 2026-06-27 blackW flat 与 obstacle 分叉结果记录

## 1. Jun27_01-04-38_ 与 Jun27_01-11-44_ 复查

`Jun27_01-04-38_` 属于 Flat 稳定性线：从 `Jun26_20-12-37_` 继续 1000 iter 到 35000，保持 `hip_default = -0.55`，将 `run_still` 从 -1.0 加到 -1.5，并使用 `roll_orientation = -1.0`。目标是继续强化平地 x/yaw/x+yaw 姿态稳定。

`Jun27_01-11-44_` 属于 Obstacle 折中线：从 `Jun26_12-23-31_` 继续 1000 iter 到 34000，温和增强为 `hip_default = -0.4`、`roll_orientation = -1.0`，保持 `run_still = -1.0`，并将 `wheel_obstacle_lift/spin.horizontal_force_threshold` 从 15 降到 10、`wheel_obstacle_lift.clearance_margin` 从 0.05 提到 0.06。目标是在保留越障能力的前提下小幅补平地稳定性和触障抬轮敏感度。

末 500 iter 相对各自父 run 看，`Jun27_01-04-38_` 延续了 Flat 线特点：`tracking_ang_vel` 从约 1.23 提到 1.30，`rew_smoothness`、`rew_torques`、`rew_action_rate` 和 `dof_pos_limits` 继续变轻；但 `terrain_level` 从约 5.39 降到 4.60，`rew_wheel_obstacle_lift` 从约 0.87 降到 0.72，`progress` 和 x tracking 也下降。这说明它更适合平地稳定性验证，不适合作为越障模型。

`Jun27_01-11-44_` 相对 `Jun26_12-23-31_` 也出现 `rew_run_still` 量级大幅变重，mean reward 因此不可直接与父 run 比；但它保留了明显更好的越障相关指标：末 500 `terrain_level` 约 5.43，`rew_wheel_obstacle_lift` 约 0.93，显著高于 Flat 线的约 4.60 和 0.72。它的 `tracking_ang_vel` 从约 1.17 提到 1.22，动作代价也比父 run 更轻，但 y tracking 与 progress 有所下降。

当前判断：`Jun27_01-04-38_` 可作为 Flat 线候选，只用于 sim2real 平地 x+yaw 稳定性观察；不应要求它保留高墙能力。`Jun27_01-11-44_` 更适合作为 Obstacle 线的下一步候选，因为它在温和稳定性增强后仍保留较好的 terrain/high-wall 指标。后续建议分别评估：Flat 线看实机平地侧翻和轮子并拢是否改善；Obstacle 线先做 sim2sim 高墙/楼梯，再决定是否进入 sim2real 越障对比。

## 2. Obstacle 线高墙无载抬轮放松实验

sim2sim 观察 `Jun27_01-11-44_` 上楼梯尚可，但高墙翻越不够丝滑。当前判断是温和稳定性约束、较早 anti-spin 以及 `wheel_obstacle_unloaded_lift = -0.08` 共同压缩了高墙动作空间。本次先做最小改动：以 `Jun27_01-11-44_` 为默认 resume 起点，保持其 `hip_default = -0.4`、`roll_orientation = -1.0`、`wheel_obstacle_lift/spin.horizontal_force_threshold = 10`、`wheel_obstacle_lift.clearance_margin = 0.06`，仅将 `wheel_obstacle_unloaded_lift` 从 -0.08 放松到 -0.06。

目标是释放一点高墙无载抬轮/身体翻越动作空间，同时比 -0.05 更保守。观察重点：高墙是否更连贯，是否仍保持楼梯抬腿与 anti-spin，动作是否变散，TensorBoard 中 `rew_wheel_obstacle_unloaded_lift`、`rew_wheel_obstacle_lift`、`rew_smoothness`、`rew_torques` 和 `value_function` 是否出现副作用。

## 3. Jun27_01-11-44_ sim2real 越障能力复盘

sim2real 观察：`Jun27_01-11-44_` 的实际越障能力仍不如 `Jun26_12-23-31_` 和 `Jun26_12-30-56_obstacle`，因此原计划从 `Jun27_01-11-44_` 继续训练的路线先搁置。

对比配置后判断，`Jun27_01-11-44_` 不是单一越障参数改动，而是同时加入了稳定性和触障触发变化：`hip_default` 从 -0.35 加到 -0.4，打开 `roll_orientation = -1.0`，`run_still` 采用 y/yaw 衰减后在 x+yaw 混合指令下持续生效，同时 `wheel_obstacle_lift/spin.horizontal_force_threshold` 从 15 降到 10、`wheel_obstacle_lift.clearance_margin` 从 0.05 提到 0.06。这些改动有利于平地/yaw 姿态和动作代价，但会压缩高墙所需的抬轮、俯仰和接触推进动作空间。

当前更可靠的 obstacle 基线仍应回到 `Jun26_12-23-31_` 或 `Jun26_12-30-56_`：前者更保守稳定，后者仅将 `wheel_obstacle_unloaded_lift` 从 -0.08 放松到 -0.05，变量干净且 sim2sim 中上楼梯抬腿更积极。后续 obstacle 线不建议继续叠加 `Jun27_01-11-44_` 的稳定性约束；应先从 6/26 obstacle 基线做单变量高墙增强，例如只调 `wheel_obstacle_unloaded_lift`、高墙 active time 或高墙 clearance，确认高墙/楼梯能力后再轻量补稳定性。

## 4. 本地配置回退到 Jun26_12-23-31_

根据 sim2real 越障复盘，先将本地默认训练配置切回 `Jun26_12-23-31_` 对应的 obstacle 保守基线：runner 从 `Jun26_09-52-36_` resume，`hip_default = -0.35`、`run_still = -1.0`、`roll_orientation = 0.0`、`wheel_obstacle_lift/spin.horizontal_force_threshold = 15`、`wheel_obstacle_lift.clearance_margin = 0.05`、`wheel_obstacle_unloaded_lift = -0.08`。同时将 `hip_default/run_still` 的 y 方向衰减参数恢复为 6/26 基线数量级 `y_ref = 0.5`、`y_scale = 0.35`。

注意：当前 env 里的 `run_still` 仍保留后续新增的 y/yaw 衰减制逻辑，本次只回退 config 参数，不修改 reward 实现。

## 5. run_still 衰减制加入配置开关

为兼顾 6/26 obstacle 基线复现和平地稳定性实验，将 `_reward_run_still` 改为由 `rewards.run_still.use_command_decay` 控制：`False` 时使用旧的 y/yaw 阈值门控，即只有 `abs(cmd_x)` 超过阈值且 `abs(cmd_y)`、`abs(cmd_yaw)` 低于阈值时才启用；`True` 时使用后续新增的 y/yaw 连续衰减制，在 x+yaw 混合指令下仍保留部分回中约束。

当前默认设为 `use_command_decay = False`，使本地配置更接近 `Jun26_12-23-31_` obstacle 基线。后续如果要单独测试平地 x+yaw 稳定增强，只需要把该开关改为 `True`，不再改 reward 代码。

## 6. Jun27_17-07-17_ 与 Jun27_17-09-43_ 结果复查

两组都从 `Jun26_12-23-31_` 继续 1000 iter 到 34000，并保持 6/26 obstacle 基线的稳定性设置：`hip_default = -0.35`、`roll_orientation = 0.0`、`run_still.use_command_decay = False`、`wheel_obstacle_lift/spin.horizontal_force_threshold = 15`。`Jun27_17-07-17_` 的改动是 `wheel_obstacle_unloaded_lift = -0.06` 且 `wheel_obstacle_spin.stairs_continuous_scale = 1.6`；`Jun27_17-09-43_` 的改动是 `wheel_obstacle_unloaded_lift = -0.06`、`wheel_obstacle_lift.clearance_margin = 0.06`、`over_lift_margin = 0.04`。

末 500 iter 均值看，两组都没有明显退化：`Jun27_17-07-17_` 的 `mean_reward` 约 72.43、`terrain_level` 约 5.63、`rew_wheel_obstacle_lift` 约 1.00；`Jun27_17-09-43_` 分别约 72.18、5.60、0.99。两组 `rew_run_still` 约 -0.07，说明当前确实回到了旧阈值制的轻量级，不再出现 `Jun27_01-11-44_` 那类 run_still 大幅变重。相对 `Jun26_12-23-31_`，`17-07-17` 的 yaw tracking、action_rate、smoothness 略好，terrain_level 略高；`17-09-43` 的 wheel spin 惩罚略轻、orientation 略好，但 `dof_pos_limits` 和 value loss 稳定性稍差。

value loss 方面两组都有孤立尖峰：`17-07-17` 在 step 33376 约 57.3，但末 500 最大约 0.57；`17-09-43` 在 step 33835 约 5.31，末 100 已恢复到约 0.31 均值。当前判断两组都不像持续崩坏。优先 sim2sim 对比：`17-07-17` 看楼梯空转是否比 `-0.06` 单纯放松更好，以及高墙是否保持；`17-09-43` 看高墙是否因更高 clearance/over-lift 更丝滑，重点注意是否出现抬轮过高、动作发飘或落地恢复变差。

sim2sim 观察：`Jun27_17-07-17_` 上楼梯有时仍会卡住，尤其在上到一半、前轮和后轮都抵着台阶时，可能出现轮子直接不转或持续空转。说明仅将 `wheel_obstacle_unloaded_lift` 放松到 -0.06 并把 `stairs_continuous_scale` 从 1.5 加到 1.6，还不足以解决楼梯中段“前后轮同时受阻”的接触节奏问题；继续加大 stairs spin 惩罚也未必可靠，因为当前现象同时包含“不转”和“空转”，不是单纯空转过多。

`Jun27_17-09-43_` 相对更容易抬腿，高墙前腿抬得更高，说明提高 `clearance_margin` 和 `over_lift_margin` 对抬腿高度确实有效。但问题没有完全消失：楼梯仍未完全杜绝卡住，高墙时后腿有时还会卡住。当前判断是前腿高度问题已有改善，剩余瓶颈更偏向后腿/最后一轮越过障碍时的推进、落地恢复和持续接触脱困，而不是单纯继续提高前腿抬轮目标。

后续分析方向：楼梯中段需要区分轮子“不转”与“空转”的成因，前者可能是姿态/关节动作空间或接触几何导致策略主动停轮，后者才是 anti-spin/progress 约束不足；高墙后腿卡住则更可能需要针对后腿越过障碍的持续进度或接触后 active window，而不是继续全局增加 lift height。下一步暂不改配置，先保留这两组作为 sim2sim 对照样本。
