# 2026-06-09 blackW 越障 target 分段调整记录

## 1. Jun08 A/B 组 sim2sim 反馈后的判断

反馈：使用 Jun08_22-07-49_ 的 model_18000 和 Jun08_22-09-55_ 的 model_16000 做 sim2sim，B 组姿态表现更好；两组越障能力都有下降，楼梯上看起来能沿台阶滚爬但还不够稳定，偶尔卡住，高墙通过率偏低。

判断：wheel_lateral_clearance 已按 wheel center 使用 ground_height + wheel_radius + target_extra_height；但 wheel_obstacle_lift 的目标高度使用 obstacle_height + clearance_margin，没有加 wheel_radius。这个偏低目标有利于楼梯上形成轮子贴边滚爬，减少过高抬腿；但遇到高墙时轮心目标高度不足，容易顶墙、卡住或空转。B 组 over-lift 惩罚更强，姿态更稳，但会进一步压低主动抬轮意愿。

## 2. 本次改动

- 保留当前 B 组较稳的 wheel_lateral_clearance 参数不动。
- wheel_obstacle_lift 新增高障碍分段：obstacle_rel_height <= 0.09m 时沿用低目标 obstacle_height + clearance_margin；超过 0.09m 时切换到 obstacle_height + wheel_radius + high_obstacle_clearance_margin。
- 新增 high_obstacle_clearance_margin = 0.02，用于高墙/高障碍时给轮心留出小余量。
- 将 over_lift_margin / sigma / penalty_weight 从 B 组强压高参数 0.015 / 0.035 / 0.8 放松到 0.02 / 0.04 / 0.5，避免高障碍 target 修正后过早惩罚必要抬轮。

预期：低台阶/楼梯继续保留滚爬倾向，不强迫大幅抬腿；高墙场景给轮心足够高度目标，提高越障机会并减少顶墙卡住。观察 rew_wheel_obstacle_lift、rew_wheel_obstacle_spin、collision、base_height、torques、terrain_level、value_loss，以及 sim2sim 中楼梯是否仍平顺、高墙是否减少卡住。

风险：高障碍分段可能重新诱导部分场景抬轮偏高；如果 value_loss 或 torques 后期抬升，需要进一步降低 wheel_obstacle_lift scale 或提高 high_obstacle_height_threshold，并考虑减少高墙 terrain 比例。


## 3. high_obstacle_height_threshold 上调

在分段 target 基础上，将 high_obstacle_height_threshold 从初始建议的 0.09 调高到 0.18。这样更多楼梯/中低障碍会继续使用 obstacle_height + clearance_margin 的低目标，保留轮子沿台阶滚爬的行为；只有更接近高墙的障碍才切换到 obstacle_height + wheel_radius + high_obstacle_clearance_margin。

预期：减少普通楼梯和较低障碍被误判为高障碍后诱导抬轮偏高，优先保护 B 组较好的姿态表现和楼梯滚爬效果。观察重点是高墙是否仍能触发足够抬轮、卡墙/空转是否下降，以及楼梯上是否比 0.09 阈值更稳定。


## 4. B 组后期 checkpoint 清理

根据 sim2sim 结果，Jun08_22-09-55_ 使用 model_16000 表现相对更可用，而 16000 之后训练曲线已进入退化阶段。已删除该 run 中 model_16050.pt 到 model_20001.pt 共 81 个后期 checkpoint，保留 model_15050.pt 到 model_16000.pt、config.json、git_metadata.json 和 TensorBoard event 文件。清理后该目录约 159M。

目的：减少无效后期模型占用，避免后续误用 16000 之后已经退化的 B 组 checkpoint。


## 5. Jun09_15-13-47_ 训练结果观察

该 run 使用 high_obstacle_height_threshold = 0.18，从 B 组可用点附近继续短训到 17000。训练曲线整体健康：mean_reward 后期约 63-66，mean_episode_length 接近 970，terrain_level 保持在约 4.7-4.8，value_loss 约 0.15-0.19，mean_noise_std 下降到约 0.268；wheel_obstacle_lift 保持正收益并在后期略有上升，wheel_obstacle_spin 维持很低。

sim2sim 反馈：越障能力确实改善，但高墙翻越能力仍不够，尤其约 0.5m/s 的低速前进下成功率很低。0.5m/s 已高于 wheel_obstacle_lift / wheel_obstacle_spin 的 0.2 command 门槛，因此当前问题不应主要归因于 command gate 未触发。更可能的原因是高墙需要提前组织抬轮和抬身，而当前 wheel_obstacle_lift 依赖水平接触力触发，往往在轮子已经顶到墙后才开启 0.8s 的抬轮窗口；高速时可以借动量滚/冲过去，0.5m/s 下动量不足，接触后再抬容易变成顶墙、空转或卡住。

后续方向：优先考虑高墙场景在接触触发后的更长抬轮/姿态调整窗口，而不是简单降低 command 门槛或使用部署侧不可观测的前方地形预触发。暂不加入额外 progress reward，先通过 active_time 随高度延长来验证低速高墙是否受限于触墙后的尝试时间。观察重点为 0.5m/s 左右 cmd_x 下高墙成功率、触墙后的抬轮持续时间、卡墙空转、torques、collision 和 value_loss。


## 6. 高障碍 active_time 随高度延长

针对 0.5m/s 左右低速高墙翻越成功率仍低的问题，本次不采用前方地形预触发，因为部署侧当前没有地形感知，提前用 height sample 给 reward 信号可能不利于学习可迁移策略。改为保留接触触发机制，只让 wheel_obstacle_lift 的 active_time 随障碍高度延长。

本次改动：新增 high_obstacle_active_height_span = 0.18 和 high_obstacle_extra_active_time = 0.4。当 obstacle_rel_height 低于 high_obstacle_height_threshold = 0.18 时仍使用基础 active_time = 0.8s；超过 0.18m 后按高度线性增加窗口，约到 0.36m 及以上时 active_time 最高为 1.2s。

预期：低台阶和楼梯仍保持短窗口滚爬，不额外诱导高抬腿；真正高墙在触发后获得更长的抬轮和姿态调整时间，改善 0.5m/s 下因动量不足导致的顶墙、空转和卡住。观察重点：低速高墙通过率、触墙后的抬轮持续时间、是否出现高墙后动作残留、torques、collision、rew_wheel_obstacle_lift 和 value_loss。

风险：高墙窗口变长可能导致越过障碍后仍保持抬轮动作，若 sim2sim 出现动作残留或落地恢复变差，可降低 high_obstacle_extra_active_time 到 0.2-0.3，或增大 high_obstacle_active_height_span 让延长更缓。


## 7. 高墙 target 余量与 active_time 进一步增强

在高障碍分段 target 和 active_time 随高度延长的基础上，进一步调高两个高墙相关参数：

- high_obstacle_clearance_margin 从 0.02 提到 0.04，高障碍时轮心目标高度从 obstacle_height + wheel_radius + 0.02 提高到 +0.04，给高墙翻越留出更多安全余量。
- high_obstacle_extra_active_time 从 0.4 提到 0.8，高障碍 active_time 从最高 1.2s 增加到最高 1.6s，给 0.5m/s 左右低速触墙后的抬轮和姿态调整更多时间。

预期：增强高墙尤其低速高墙的翻越能力，减少因轮心高度不足或抬轮窗口过短导致的顶墙卡住。风险是高墙后动作残留、抬轮偏高或 torques 上升；下一轮重点观察低速高墙成功率、越墙后的落地恢复、rew_wheel_obstacle_lift、collision、torques 和 value_loss。


## 8. 高墙地形整块周期铺设

观察：当前高墙地形只在每个地形块机器人出生位置前放置一组高墙。机器人翻越当前地形块的高墙并进入下一个地形块后，会先经历较长平地，再遇到下一组高墙，导致 episode 后半段训练信号被平地速度跟踪稀释，高墙连续处理能力不足。

本次采用保守方案：为 high_wall_terrain 增加 fill_full_block / spawn_clearance 参数。blackW 中开启 high_wall_fill_full_block = True，并设置 high_wall_spawn_clearance = 0.8。高墙仍保留原来出生点前方 start_distance = 1.2、spacing = 1.2 的相位，同时在出生点后方镜像铺设，跳过中心出生点附近 0.8m 范围。对于 8m 地形块，典型相对位置为 -3.6、-2.4、-1.2、1.2、2.4、3.6m。

预期：跨入下一个高墙地形块后更快再次遇到高墙，减少长平地段对高墙训练信号的稀释；同时出生点附近仍保持安全空区，避免 reset 后直接卡墙。观察重点：高墙连续翻越成功率、terrain_level、episode length、collision、torques、低速高墙表现，以及是否因墙密度增加导致训练不稳定。
