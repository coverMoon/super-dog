# AGENTS.md

## 项目背景

本仓库用于 HIMLoco / legged_gym 轮足与四足机器人训练实验。当前重点任务是 `blackW` 轮足机器人在 sim2sim 和 sim2real 中的速度跟踪、课程学习、reward shaping 和部署稳定性调参。

主要实验代码位于：

- `legged_gym/legged_gym/envs/blackW/`
- `legged_gym/legged_gym/envs/black/`
- `legged_gym/legged_gym/envs/base/`
- `rsl_rl/rsl_rl/runners/`

实验记录按机器人分目录保存：

- `日志/blackW/`：轮足 `blackW` 实验记录
- `日志/black/`：点式足 `black` 实验记录

修改重要 reward、curriculum、domain randomization、resume 行为后，应在对应机器人目录下新增中文记录。注意根据日期来补充并按日期来归类，每个机器人每天只保留一个文件，若当日还没有日志，可自行新建一个。在记录时根据改动内容大小适量记录，不要过于冗长。

## 机器人上下文选择

处理任务前先判断当前机器人上下文，只阅读对应目录的日志，不要同时读取 `black` 和 `blackW` 两套日志。判断优先级如下：

- 用户明确提到 `black` 或 `blackW` 时，以用户指定为准。
- 用户提到具体代码路径时，`envs/blackW/` 对应 `blackW`，`envs/black/` 对应 `black`。
- 用户提到训练 run 时，优先从 run 所在日志目录、任务名或配置中的 `experiment_name` 推断机器人。
- 若本轮没有新线索，则沿用当前对话中最近一次明确的机器人上下文。
- 若仍无法判断，先问一句确认，不要为了保险同时读两边日志。

## 工作习惯

修改代码前先阅读相关日志，理解当前实验脉络。不要只看当前 diff 做判断。

优先保持改动小而明确。不要顺手重构无关模块。

当前环境中 `apply_patch` 经常因 sandbox / namespace 限制失败。需要编辑文件时，不要先试 `apply_patch`；直接使用已经验证可行的受控写入方式，例如小范围 Python 文本替换或其它明确、可复核的命令，并在改后用 `git diff` 核对。

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
```

注意：直接 import 任务配置可能因为缺少 isaacgym 环境失败，这不一定代表语法或代码逻辑错误。
