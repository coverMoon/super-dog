# 2026-06-21 blackW 上楼梯 run_still 分层记录

## 1. Jun20 双分支结果

对比 Jun20_21-26-29_（run_still = -1、difficult_posture_hold = -1）与 Jun20_21-27-59_（run_still = -3、关闭 difficult_posture_hold），sim2sim 观察上后者表现更好。TensorBoard 末 1000 iter 反推的 run_still 原始关节偏移由前者约 0.0359 降到后者约 0.0131，下降约 63%；前者的 difficult_posture_hold 贡献仍只有约 -0.00162，说明带 terrain、pitch、deadzone 和前腿 lift 权重的多层门控没有形成足够强的姿态约束。

两组 mean_episode_length 与 terrain_level 基本相同。run_still = -3 组的 progress、速度跟踪和 wheel_obstacle_lift 略低，action_rate 惩罚略重，但 smoothness 与 torques 更稳定；另一组在约 44890 附近出现 value loss、smoothness、torques 和 reward 的同步尖峰。因此当前更认可直接关节回中逻辑，但需要避免将 -3 全局施加到平地直行。

## 2. 删除 difficult_posture_hold，增加上楼梯叠加项

完整删除 difficult_posture_hold 的配置、reward 实现及其 front/rear sagittal DOF 索引。将全局 run_still 恢复为 -1.0，并新增 stairs_run_still = -2.0。

stairs_run_still 与 run_still 使用完全相同的逻辑：对全部非轮关节相对默认位置的 L1 偏移求和，并沿用 abs(cmd_x) > 0.1、abs(cmd_y) < 0.1、abs(cmd_yaw) < 0.15 的门控；额外要求地形类型为上楼梯 type 3。由此平地及其他地形保持 run_still = -1，上楼梯时两项叠加为等效 -3，复现表现较好分支的楼梯姿态约束，同时降低对平地直行的影响。

预期观察：TensorBoard 新增 rew_stairs_run_still。重点比较上楼梯前倾、关节偏移和起抬稳定性，同时关注 rew_run_still、rew_stairs_run_still、rew_wheel_obstacle_lift、rew_progress、tracking_lin_vel_x、rew_action_rate、rew_smoothness、rew_torques、terrain_level 与 mean_episode_length。若楼梯效果弱于全局 -3，说明全地形训练形成的共享姿态先验本身有贡献；若平地变自然且楼梯保持改善，则保留分层方案。

验证：已运行 python -m py_compile 检查 blackW_config.py 与 blackW_env.py，语法通过。
