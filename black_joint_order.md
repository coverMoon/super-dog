# Black / BlackW Runtime Joint Order

这份说明以 Isaac Gym 运行时打印出来的 `self.dof_names` 为准，而不是单看 URDF 文本里的关节书写顺序。

当前 `black` 和 `blackW` 在 Isaac Gym 中的运行时腿顺序一致，都是：

```text
FL -> FR -> RL -> RR
```

## 1. black 策略输出顺序

```text
0  FL_hip_joint
1  FL_thigh_joint
2  FL_calf_joint
3  FR_hip_joint
4  FR_thigh_joint
5  FR_calf_joint
6  RL_hip_joint
7  RL_thigh_joint
8  RL_calf_joint
9  RR_hip_joint
10 RR_thigh_joint
11 RR_calf_joint
```

## 2. blackW 策略输出顺序

```text
0  FL_hip_joint
1  FL_thigh_joint
2  FL_calf_joint
3  FL_wheel_joint
4  FR_hip_joint
5  FR_thigh_joint
6  FR_calf_joint
7  FR_wheel_joint
8  RL_hip_joint
9  RL_thigh_joint
10 RL_calf_joint
11 RL_wheel_joint
12 RR_hip_joint
13 RR_thigh_joint
14 RR_calf_joint
15 RR_wheel_joint
```

## 3. black 到 blackW 的部署对应关系

如果现有实机部署程序已经能正确下发 `black` 的 12 个腿关节，那么迁移到 `blackW` 时应保持腿顺序不变，只需在每条腿后插入轮子电机：

```text
FL: hip, thigh, calf        -> hip, thigh, calf, wheel
FR: hip, thigh, calf        -> hip, thigh, calf, wheel
RL: hip, thigh, calf        -> hip, thigh, calf, wheel
RR: hip, thigh, calf        -> hip, thigh, calf, wheel
```

也就是说，应按下面的腿顺序部署：

```text
FL, FR, RL, RR
```

而不是：

```text
FL, FR, RR, RL
```

## 4. 当前代码里的影响

- `black` 训练主链路按 Isaac Gym 运行时 `self.dof_names` 组装 `default_dof_pos`、PD 参数和动作，因此内部是一致的。
- `blackW` 的 `wheel_indices`、`hip_indices`、`leg_dof_indices` 都是从运行时 `self.dof_names` 动态提取的。
- `black` 的脚相关奖励是按脚名 `FL/FR/RL/RR` 建立映射，不依赖“后腿谁在前谁在后”的硬编码顺序。
- `blackW` 的 `wheel_forward_sign` 现在按腿名解析，不再依赖 `[FL, FR, RL, RR]` 这样的手写列表顺序。

## 5. 建议

- 部署侧统一以启动仿真时打印出的 `dof_names` 为准做映射。
- 如果实机控制程序里仍保存着 `FL, FR, RR, RL` 的后腿顺序假设，需要改成 `FL, FR, RL, RR`。
