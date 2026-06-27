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
