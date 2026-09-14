# Black With Arm Draft

This folder contains a first merged URDF asset for `black + arm` training experiments.

Direct decisions made in this draft:
- Arm is mounted to `trunk` through `trunk_to_arm_mount`.
- Initial mount pose is `xyz = [0.06, 0.0, 0.028]`, `rpy = [0.0, 0.0, 0.0]`.
- Arm links and joints are prefixed with `arm_` to avoid naming ambiguity.
- Arm meshes are copied locally into `meshes_arm/`.
- Arm joint `effort` and `velocity` limits use placeholder training values: `12.0` and `3.0`.

Suggested first default arm pose for later config wiring:
- `arm_yaw_joint = 0.0`
- `arm_pitch_1_joint = 1.20`
- `arm_pitch_2_joint = -2.10`
- `arm_pitch_3_joint = 0.90`
- `arm_roll_joint = 0.0`

What still needs manual tuning:
- Mount `xyz/rpy`
- Collision simplification strategy for Isaac Gym training
- Real actuator limits if the model is later aligned with hardware control
