import numpy as np
import pandas as pd


def calculate_terrain_weights(sigma, original_scale, clip_min=0.1):
    """
    计算不同地形下的姿态惩罚权重

    Args:
        sigma (float): 敏感度参数，控制权重衰减的快慢。值越大，对地形起伏越不敏感（衰减越慢）。
        original_scale (float): 原始的姿态奖励权重（通常为负值，表示惩罚）。
        clip_min (float): 权重的最小缩放比例（默认0.1），防止完全失去约束。
    """

    # 1. 定义采样点 (与 black_config.py 中的 measured_points 保持一致)
    measured_points_x = np.array(
        [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    measured_points_y = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])

    # 生成网格点 (模拟机器人周围的感知范围)
    X, Y = np.meshgrid(measured_points_x, measured_points_y)
    X_flat = X.flatten()

    # 定义地形场景列表
    scenarios = []

    # (1) 平地
    scenarios.append({
        "name": "Flat Ground (平地)",
        "heights": np.zeros_like(X_flat)
    })

    # (2) 斜坡 (Slope)
    # 假设机器人正对着坡爬，高度 z = x * tan(angle)
    slopes = [10, 15, 20, 25, 30]  # degrees
    for deg in slopes:
        rad = np.deg2rad(deg)
        heights = X_flat * np.tan(rad)
        scenarios.append({
            "name": f"Slope {deg}° (斜坡)",
            "heights": heights
        })

    # (3) 楼梯 (Stairs)
    # 模型：z = floor(x / run) * rise
    stairs_config = [
        (0.30, 0.15, "Stairs Normal (15cm/30cm)"),  # 标准楼梯
        (0.28, 0.18, "Stairs Steep (18cm/28cm)"),  # 较陡
        (0.25, 0.20, "Stairs Extreme (20cm/25cm)")  # 极限
    ]
    for run, rise, label in stairs_config:
        heights = np.floor(X_flat / run) * rise
        scenarios.append({
            "name": label,
            "heights": heights
        })

    # 2. 计算并输出结果
    results = []
    print(f"\n{'=' * 60}")
    print(f" 参数设置: Sigma = {sigma} | Original Scale = {original_scale}")
    print(f"{'=' * 60}")
    print(f"{'Terrain Type':<25} | {'STD':<8} | {'Scale':<8} | {'Final Weight':<12}")
    print(f"{'-' * 25} | {'-' * 8} | {'-' * 8} | {'-' * 12}")

    for sc in scenarios:
        # 计算该地形下采样点高度的标准差
        std = np.std(sc["heights"])

        # 核心公式: scale = exp(- std^2 / sigma)
        raw_scale = np.exp(- (std ** 2) / sigma)

        # 限制最小权重 (Clip)
        final_scale = np.clip(raw_scale, clip_min, 1.0)

        # 计算最终应用到奖励函数上的权重
        final_weight = original_scale * final_scale

        print(f"{sc['name']:<25} | {std:.4f}   | {final_scale:.4f}   | {final_weight:.4f}")


# ==========================================
# 在这里调整你的参数进行测试
# ==========================================
if __name__ == "__main__":
    # 推荐参数组合 1: 比较激进的放松，适合很难爬楼梯的情况
    # SIGMA = 0.15
    # ORIGINAL_SCALE = -4.0

    # 推荐参数组合 2: 比较保守，保留较多约束
    SIGMA = 0.03
    ORIGINAL_SCALE = -3.0

    calculate_terrain_weights(SIGMA, ORIGINAL_SCALE)