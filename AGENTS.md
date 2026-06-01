# AGENTS.md

## 项目背景

本仓库用于 HIMLoco / legged_gym 轮足与四足机器人训练实验。当前重点任务是 `blackW` 轮足机器人在 sim2sim 和 sim2real 中的速度跟踪、课程学习、reward shaping 和部署稳定性调参。

主要实验代码位于：

- `legged_gym/legged_gym/envs/blackW/`
- `legged_gym/legged_gym/envs/black/`
- `legged_gym/legged_gym/envs/base/`
- `rsl_rl/rsl_rl/runners/`

实验记录位于：

- `日志/`

修改重要 reward、curriculum、domain randomization、resume 行为后，应在 `日志/` 下新增中文记录。注意根据日期来补充并按日期来归类，每个日期只保留一个文件，若当日还没有日志，可自行新建一个。在记录时根据改动内容大小适量记录，不要过于冗长。

## 工作习惯

修改代码前先阅读相关日志，理解当前实验脉络。不要只看当前 diff 做判断。

优先保持改动小而明确。不要顺手重构无关模块。

如果涉及训练行为变化，优先说明：

- 改动目标
- 涉及 reward / config / env 逻辑
- 预期观察指标
- 风险和回退建议

## 验证命令

常规语法检查：

```bash
python -m py_compile \
  legged_gym/legged_gym/envs/blackW/blackW_config.py \
  legged_gym/legged_gym/envs/blackW/blackW_env.py

注意：直接 import 任务配置可能因为缺少 isaacgym 环境失败，这不一定代表语法或代码逻辑错误。