# HIMLoco for Black Quadruped and BlackW

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
- `black_joint_order.md`：`black` / `blackW` 在 Isaac Gym 运行时的关节顺序与部署映射说明。
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

推荐优先使用仓库自带的 Docker 环境。Docker 和 Conda 安装方式都建议使用以下核心软件栈：

- Ubuntu 20.04（Docker 基础镜像；宿主机版本可以不同）
- Python 3.8.20
- PyTorch 1.13.1 + cu117
- torchvision 0.14.1 + cu117
- torchaudio 0.13.1 + cu117
- NumPy 1.23.5
- Matplotlib 3.7.5
- TensorBoard 2.14.0
- 本地安装 `isaacgym`、`rsl_rl`、`legged_gym`

这里的 CUDA 版本指 PyTorch 自带的 CUDA 11.7 runtime。系统安装的 `nvcc` 版本可以不同，检查训练运行时版本应使用：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 方式一：Docker

```bash
docker build -t himloco:train .
./run_train_docker.sh himloco:train
```

如果宿主机配置了 `DISPLAY`，脚本会自动挂载 X11 以支持可视化。

### 方式二：Conda

在仓库根目录执行以下命令创建 `himloco` 环境并安装依赖：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n himloco -c conda-forge python=3.8.20 pip=24.3.1 -y
conda activate himloco

python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  torch==1.13.1+cu117 \
  torchvision==0.14.1+cu117 \
  torchaudio==0.13.1+cu117 \
  numpy==1.23.5 \
  scipy==1.10.1 \
  matplotlib==3.7.5 \
  tensorboard==2.14.0 \
  protobuf==3.20.3 \
  PyYAML==6.0.3

python -m pip install -e isaacgym/python
python -m pip install -e rsl_rl
python -m pip install -e legged_gym

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

`LD_LIBRARY_PATH` 中需要优先包含当前环境的 `lib/`，否则 Isaac Gym 的 `gym_38.so` 可能无法找到 `libpython3.8.so.1.0`。如需长期使用，可以将这条设置写入该 Conda 环境的激活钩子。

安装完成后执行基础检查：

```bash
python -m pip check
python -c "import isaacgym, torch, legged_gym, rsl_rl; print('imports ok')"
```

## 训练与评估

训练入口：

```bash
python legged_gym/legged_gym/scripts/train.py --task=black
python legged_gym/legged_gym/scripts/train.py --task=blackW
```

策略回放：

```bash
python legged_gym/legged_gym/scripts/play.py --task=black
python legged_gym/legged_gym/scripts/play.py --task=blackW
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
