# blackW 课程学习 resume 状态持久化记录

日期：2026-05-28

## 背景

之前从 checkpoint resume 训练时，策略网络和优化器会恢复，但 command curriculum 的运行时状态不会恢复。结果是 `lin_vel_x`、`lin_vel_y`、`ang_vel_yaw` 的课程范围会回到 config 初始值，导致续训时难度突然降低，TensorBoard 里的课程进度也会看起来像被重置。

但完全恢复课程统计也不理想：如果 resume 前后修改了 reward 权重、tracking 阈值、command 采样概率、控制方式或 domain randomization，旧的 EMA / pass streak 已经不能代表新配置下的表现。

## 当前方案

采用中间方案：默认只恢复课程范围，重新估计统计量。

新增 runner 配置：

```python
resume_command_curriculum = 'range'
```

可选值：

```text
none   完全不恢复课程状态，使用 config 初始范围
range  默认，只恢复 command range，重置 EMA、buffer、pass streak、score
full   恢复 command range、EMA、pass streak；只建议完全同配置续训时使用
```

## 保存规则

为了避免每个 checkpoint 都生成额外 JSON，占用空间过多，课程状态只在两个场景保存：

1. 正常训练结束时保存最终模型
2. 训练被 Ctrl+C 打断时保存 interrupt 模型

中间周期性 checkpoint 不保存课程 JSON。

示例：

```text
model_500.pt
command_curriculum_500.json

interrupt_model_827.pt
interrupt_command_curriculum_827.json
```

中间 checkpoint 示例：

```text
model_20.pt
model_40.pt
model_60.pt
```

这些不会生成对应的课程 JSON。

## JSON 内容

课程 JSON 记录三类信息：

```json
{
  "version": 1,
  "iteration": 500,
  "command_ranges": {
    "lin_vel_x": [-1.4, 1.4],
    "lin_vel_y": [-0.6, 0.6],
    "ang_vel_yaw": [-1.6, 1.6]
  },
  "ema": {
    "x_low": 0.58,
    "x_high": 0.51,
    "y_low": 0.42,
    "y_high": 0.36,
    "yaw_low": 0.33,
    "yaw_high": 0.29
  },
  "pass_streak": {
    "x": 1,
    "y": 0,
    "yaw": 0
  }
}
```

默认 `range` 模式只读取 `command_ranges`。`ema` 和 `pass_streak` 会保存在文件里，但不会默认恢复。

## resume 行为

加载 `model_500.pt` 时，会自动寻找同目录下的：

```text
command_curriculum_500.json
```

加载 `interrupt_model_827.pt` 时，会自动寻找：

```text
interrupt_command_curriculum_827.json
```

如果找到 JSON：

- `range` 模式恢复 x/y/yaw 的课程范围
- 清空 EMA、buffer、pass streak、score
- 清空当前 command segment 累积
- 立刻重新采样所有环境的 command，保证第一批 rollout 使用恢复后的课程范围

如果找不到 JSON：

- 打印提示
- 使用 config 中的初始课程范围继续训练

## 相关代码

- `rsl_rl/rsl_rl/runners/him_on_policy_runner.py`
  - checkpoint 保存时决定是否写课程 JSON
  - resume 时按模型文件名寻找课程 JSON

- `legged_gym/legged_gym/envs/blackW/blackW_env.py`
  - `get_command_curriculum_state()` 导出课程状态
  - `load_command_curriculum_state()` 恢复课程状态
  - `_reset_command_curriculum_statistics()` 重置 EMA、buffer、pass streak、score

- `legged_gym/legged_gym/envs/base/legged_robot_config.py`
  - 新增 `resume_command_curriculum`

## 使用建议

常规调参续训使用默认值：

```python
resume_command_curriculum = 'range'
```

只有在确认 reward、采样概率、阈值、控制方式等都没有变化，只是单纯接着同一个实验跑时，才考虑：

```python
resume_command_curriculum = 'full'
```

如果希望完全重新开始课程学习，但仍加载策略权重，可以设为：

```python
resume_command_curriculum = 'none'
```
