# HIMLoco for Black Quadruped and Black-Arm

本仓库基于 [HIMLoco](./projects/himloco/README.md)、[legged_gym](./legged_gym/README.md) 和 `rsl_rl` 做了面向 `black` 四足机器人任务的工程化改造，当前重点支持以下三类训练场景：

- `black`：面向自定义 `black` 四足机器人本体的粗糙地形运动控制。
- `black_bridge`：面向断桥/窄桥通过任务的专用课程与奖励设计。
- `black_arm`：在 `black` 机体上集成机械臂后的 locomotion + manipulation 联合训练环境。

根目录 README 主要介绍当前工程的可用内容；原始论文材料仍保留在 [projects/himloco](./projects/himloco/README.md) 和 [projects/h_infinity](./projects/h_infinity/README.md)。

## 项目概览

相对原始代码，这个仓库的核心改动集中在 `black` 机器人资产、任务环境和训练配置上：

- 新增 `black` 机器人 URDF 与网格资源，接入 `legged_gym` 训练管线。
- 新增 `black`、`black_bridge`、`black_arm` 三个任务注册入口。
- 针对 `black` 任务重新配置了 PD 参数、动作缩放、命令课程、地形分布和奖励权重。
- 在 domain randomization 中加入了负载质量、质心偏移、连杆质量、摩擦、驱动强度、KP/KD、惯量和动作延迟等扰动。
- 为桥面任务增加了面向“跨 gap / 防卡边 / 快速恢复”的专用奖励项与桥面课程参数。
- 新增 `black_with_arm` 组合 URDF、机械臂轨迹库、负载吸附/释放模拟，以及 locomotion-only / stationary manipulation / carry-and-move 三种模式切换。
- 保留了 HIMLoco 与 H-infinity 相关论文材料和 citation，但当前 `legged_gym` 默认训练主线仍是 HIM 系列实现：`HIMOnPolicyRunner + HIMPPO + HIMActorCritic`。

## 主要目录

- `legged_gym/legged_gym/envs/black/`：`black` 与 `black_bridge` 环境及配置。
- `legged_gym/legged_gym/envs/black_arm/`：机械臂轨迹库、联合任务环境与配置。
- `legged_gym/resources/robots/black/`：`black` 机器人资产。
- `legged_gym/resources/robots/black_with_arm/`：`black + arm` 组合资产。
- `robust_arm/`：机械臂原始模型、任务规划草案和实验数据。
- `terrain_weights_calc.py`：辅助分析地形自适应奖励权重的小工具。
- `projects/himloco/`：HIMLoco 论文主页说明。
- `projects/h_infinity/`：H-infinity locomotion 论文主页说明。

## 当前任务说明

### `black`

`black` 是当前最基础的 locomotion 任务，面向粗糙地形行走。它在原始 `legged_gym` 风格任务之上做了几类强化：

- 使用自定义 `black_description.urdf` 与对应的关节默认位姿、PD 参数。
- 引入命令课程学习与较宽的速度/航向命令范围。
- 引入更强的 domain randomization，包括外力扰动、推搡、负载质量变化和动作延迟。
- 使用地形相关的奖励调节逻辑，减轻机器人在坡面/台阶等复杂地形上的训练不稳定问题。

### `black_bridge`

`black_bridge` 是从 `black` 派生出的断桥专项任务，重点不是泛化粗糙地形，而是提升跨 gap 稳定性：

- 地形分布大幅偏向桥面课程。
- 增加桥缝宽度、木板长度、坑深等专项课程参数。
- 定制了 `gap_clearance`、`gap_recovery_burst`、`edge_escape` 等奖励，针对抬腿跨缝、失误后恢复、后腿卡边等典型失败模式做优化。

### `black_arm`

`black_arm` 在四足本体上集成了机械臂与末端负载模拟，用于 locomotion 与 manipulation 结合场景：

- 组合资产使用 `black_with_arm_train.urdf`。
- 机械臂轨迹采用脚本库驱动，当前内置 `grasp / transfer / place / lift_hold` 等原语。
- 支持 `locomotion_only`、`manip_stationary`、`carry_move` 三类任务模式采样。
- 支持末端负载质量、负载偏移与时序扰动模拟。
- 当前策略动作维度仍对应 12 个腿部关节，机械臂主要通过参考轨迹和 PD 控制跟踪。

## 环境准备

推荐优先使用仓库自带的 Docker 环境，当前 `Dockerfile` 固定了：

- Ubuntu 20.04
- Python 3.8
- CUDA 11.7
- PyTorch 1.13.1 + cu117
- 本地安装 `isaacgym`、`rsl_rl`、`legged_gym`

### 方式一：Docker

```bash
docker build -t himloco:train .
./run_train_docker.sh himloco:train
```

如果本机配置了 `DISPLAY`，脚本会自动挂载 X11 以支持可视化。

### 方式二：手动安装

请尽量与 Docker 中的版本保持一致，然后在仓库根目录执行：

```bash
pip install -e isaacgym/python
pip install -e rsl_rl
pip install -e legged_gym
```

## 训练与评估

训练入口：

```bash
python legged_gym/legged_gym/scripts/train.py --task=black
python legged_gym/legged_gym/scripts/train.py --task=black_bridge
python legged_gym/legged_gym/scripts/train.py --task=black_arm
```

策略回放：

```bash
python legged_gym/legged_gym/scripts/play.py --task=black
python legged_gym/legged_gym/scripts/play.py --task=black_bridge
python legged_gym/legged_gym/scripts/play.py --task=black_arm
```

日志默认保存在：

- `legged_gym/logs/rough_black_dog`
- `legged_gym/logs/bridge_black_dog`
- `legged_gym/logs/rough_black_arm`

`play.py` 默认会按实验名加载最新一次运行的最新 checkpoint。

## 相关论文与 Citation

如果你的工作使用了本仓库中的 HIM 训练框架、任务设计或论文材料，建议优先引用以下工作。

### HIMLoco

```bibtex
@inproceedings{long2023him,
  title={Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response},
  author={Long, Junfeng and Wang, ZiRui and Li, Quanyi and Cao, Liu and Gao, Jiawei and Pang, Jiangmiao},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

原始项目介绍可参考：

- [projects/himloco/README.md](./projects/himloco/README.md)
- [projects/h_infinity/README.md](./projects/h_infinity/README.md)

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

请注意，本仓库同时包含基于上游项目继承而来的代码与资源，使用时还需分别遵守对应子目录中的原始许可证与声明。

## Acknowledgements

- [legged_gym](https://github.com/leggedrobotics/legged_gym): The training codebase is built on top of legged_gym.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl): PPO and related RL infrastructure.
- NVIDIA Isaac Gym: GPU-accelerated simulation backend used by this project.
