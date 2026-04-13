import argparse
import math
import numpy as np


UNDER_BODY_POINTS_X = np.array(
    [-0.30, -0.24, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18, 0.24, 0.30],
    dtype=float,
)
UNDER_BODY_POINTS_Y = np.array(
    [-0.18, -0.135, -0.09, -0.045, 0.0, 0.045, 0.09, 0.135, 0.18],
    dtype=float,
)
DEFAULT_STEP_WIDTH = 0.30
DEFAULT_CLIP = 0.30
DEFAULT_MAX_SCALE = 1.00
DEFAULT_STEP_HEIGHTS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.23]
DEFAULT_NUM_OFFSETS = 30
BASE_STANCE_TOLERANCE = 0.02

TERM_CONFIGS = {
    "orientation": {
        "kind": "decay",
        "sigma": 0.009,
        "min_scale": 0.10,
        "max_scale": 1.00,
        "original_scale": -3.0,
        "adaptive_label": "pitch_scale",
        "rew_label": "pitch_rew_scale",
        "formula": "final pitch reward scale = original_orientation_scale * pitch_scale",
    },
    "action_rate": {
        "kind": "decay",
        "sigma": 0.01,
        "min_scale": 0.20,
        "max_scale": 1.00,
        "original_scale": -0.3,
        "adaptive_label": "action_rate_scale",
        "rew_label": "action_rate_rew_scale",
        "formula": "final action_rate reward scale = original_action_rate_scale * action_rate_scale",
    },
    "smoothness": {
        "kind": "decay",
        "sigma": 0.2,
        "min_scale": 0.90,
        "max_scale": 1.00,
        "original_scale": -0.01,
        "adaptive_label": "smoothness_scale",
        "rew_label": "smoothness_rew_scale",
        "formula": "final smoothness reward scale = original_smoothness_scale * smoothness_scale",
    },
    "foot_clearance": {
        "kind": "margin",
        "std_gain": 2.0,
        "max_extra_clearance": 0.15,
        "stance_gain": 0.5,
        "swing_high_penalty_weight": 0.25,
        "formula": "extra_clearance = clamp(std_gain * terrain_variability, 0, max_extra_clearance)",
    },
}


def make_under_body_grid():
    grid_x, grid_y = np.meshgrid(UNDER_BODY_POINTS_X, UNDER_BODY_POINTS_Y)
    return grid_x.flatten(), grid_y.flatten()


def adaptive_scale(terrain_variability, sigma, clip_value, min_scale, max_scale):
    tv_used = min(max(terrain_variability, 0.0), clip_value)
    raw_scale = math.exp(-(tv_used ** 2) / sigma)
    final_scale = min(max(raw_scale, min_scale), max_scale)
    return tv_used, raw_scale, final_scale


def estimate_stair_heights(x_points, step_width, step_height, offset):
    levels = np.floor((x_points + offset) / step_width)
    return levels * step_height


def evaluate_decay_case(
    x_points,
    step_width,
    step_height,
    sigma,
    clip_value,
    min_scale,
    max_scale,
    original_scale,
    num_offsets,
):
    offsets = np.linspace(0.0, step_width, num_offsets, endpoint=False)
    rows = []
    for offset in offsets:
        heights = estimate_stair_heights(x_points, step_width, step_height, offset)
        std = float(np.std(heights))
        tv_used, raw_scale, adaptive = adaptive_scale(std, sigma, clip_value, min_scale, max_scale)
        rows.append(
            {
                "offset": float(offset),
                "std": std,
                "tv_used": tv_used,
                "raw_scale": raw_scale,
                "adaptive_scale": adaptive,
                "rew_scale": original_scale * adaptive,
            }
        )
    return rows


def evaluate_foot_clearance_case(
    x_points,
    step_width,
    step_height,
    clip_value,
    std_gain,
    max_extra_clearance,
    stance_gain,
    swing_high_penalty_weight,
    num_offsets,
):
    offsets = np.linspace(0.0, step_width, num_offsets, endpoint=False)
    rows = []
    for offset in offsets:
        heights = estimate_stair_heights(x_points, step_width, step_height, offset)
        std = float(np.std(heights))
        tv_used = min(max(std, 0.0), clip_value)
        raw_extra_clearance = std_gain * tv_used
        extra_clearance = min(max(raw_extra_clearance, 0.0), max_extra_clearance)
        stance_tolerance = BASE_STANCE_TOLERANCE + stance_gain * extra_clearance
        rows.append(
            {
                "offset": float(offset),
                "std": std,
                "tv_used": tv_used,
                "raw_extra_clearance": raw_extra_clearance,
                "extra_clearance": extra_clearance,
                "stance_tolerance": stance_tolerance,
                "swing_high_penalty_weight": swing_high_penalty_weight,
            }
        )
    return rows


def summarize_decay_rows(rows):
    best_relaxed = min(rows, key=lambda row: row["adaptive_scale"])
    least_relaxed = max(rows, key=lambda row: row["adaptive_scale"])
    mean_adaptive_scale = float(np.mean([row["adaptive_scale"] for row in rows]))
    mean_rew_scale = float(np.mean([row["rew_scale"] for row in rows]))
    mean_std = float(np.mean([row["std"] for row in rows]))
    return best_relaxed, least_relaxed, mean_adaptive_scale, mean_rew_scale, mean_std


def summarize_foot_clearance_rows(rows):
    most_relaxed = max(rows, key=lambda row: row["extra_clearance"])
    least_relaxed = min(rows, key=lambda row: row["extra_clearance"])
    mean_std = float(np.mean([row["std"] for row in rows]))
    mean_extra_clearance = float(np.mean([row["extra_clearance"] for row in rows]))
    mean_stance_tolerance = float(np.mean([row["stance_tolerance"] for row in rows]))
    return most_relaxed, least_relaxed, mean_std, mean_extra_clearance, mean_stance_tolerance


def print_decay_term_table(term_name, step_heights, step_width, clip_value, num_offsets, sigma, min_scale, max_scale, original_scale):
    x_points, _ = make_under_body_grid()
    term_cfg = TERM_CONFIGS[term_name]

    print(f"\nCurrent {term_name} adaptive model")
    print("adaptive_scale = clamp(exp(-(terrain_variability^2) / sigma), min_scale, max_scale)")
    print(term_cfg["formula"])
    print(
        "terrain_variability is computed from under-body height std over "
        f"{len(UNDER_BODY_POINTS_X)} x {len(UNDER_BODY_POINTS_Y)} samples"
    )
    print(
        f"step_width = {step_width:.3f} m, clip = {clip_value:.3f}, sigma = {sigma:.4f}, "
        f"min_scale = {min_scale:.3f}, max_scale = {max_scale:.3f}"
    )
    print(f"original {term_name} scale = {original_scale:.4f}")

    header = (
        f"{'step_h':>8} | {'std_max':>8} | {'scale_min':>9} | {'rew_min':>10} | "
        f"{'std_mean':>8} | {'scale_mean':>10} | {'rew_mean':>10} | {'offset@min':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for step_height in step_heights:
        rows = evaluate_decay_case(
            x_points,
            step_width,
            step_height,
            sigma,
            clip_value,
            min_scale,
            max_scale,
            original_scale,
            num_offsets,
        )
        best_relaxed, _, mean_adaptive_scale, mean_rew_scale, mean_std = summarize_decay_rows(rows)
        print(
            f"{step_height:8.3f} | {best_relaxed['std']:8.4f} | {best_relaxed['adaptive_scale']:9.4f} | "
            f"{best_relaxed['rew_scale']:10.4f} | {mean_std:8.4f} | {mean_adaptive_scale:10.4f} | "
            f"{mean_rew_scale:10.4f} | {best_relaxed['offset']:10.4f}"
        )


def print_decay_term_detail(term_name, step_height, step_width, clip_value, num_offsets, sigma, min_scale, max_scale, original_scale):
    x_points, _ = make_under_body_grid()
    term_cfg = TERM_CONFIGS[term_name]
    rows = evaluate_decay_case(
        x_points,
        step_width,
        step_height,
        sigma,
        clip_value,
        min_scale,
        max_scale,
        original_scale,
        num_offsets,
    )
    best_relaxed, least_relaxed, mean_adaptive_scale, mean_rew_scale, mean_std = summarize_decay_rows(rows)

    print(f"\nDetailed {term_name} scan for step height = {step_height:.3f} m")
    print(
        f"mean std = {mean_std:.4f}, mean {term_cfg['adaptive_label']} = {mean_adaptive_scale:.4f}, "
        f"mean {term_cfg['rew_label']} = {mean_rew_scale:.4f}"
    )
    print(
        "most relaxed offset: "
        f"offset = {best_relaxed['offset']:.4f}, std = {best_relaxed['std']:.4f}, "
        f"tv_used = {best_relaxed['tv_used']:.4f}, raw_scale = {best_relaxed['raw_scale']:.4f}, "
        f"{term_cfg['adaptive_label']} = {best_relaxed['adaptive_scale']:.4f}, "
        f"{term_cfg['rew_label']} = {best_relaxed['rew_scale']:.4f}"
    )
    print(
        "least relaxed offset: "
        f"offset = {least_relaxed['offset']:.4f}, std = {least_relaxed['std']:.4f}, "
        f"tv_used = {least_relaxed['tv_used']:.4f}, raw_scale = {least_relaxed['raw_scale']:.4f}, "
        f"{term_cfg['adaptive_label']} = {least_relaxed['adaptive_scale']:.4f}, "
        f"{term_cfg['rew_label']} = {least_relaxed['rew_scale']:.4f}"
    )

    print("\noffset scan")
    print(
        f"{'offset':>8} | {'std':>8} | {'tv_used':>8} | {'raw_scale':>10} | "
        f"{'scale':>11} | {'rew_scale':>10}"
    )
    print("-" * 74)
    for row in rows:
        print(
            f"{row['offset']:8.4f} | {row['std']:8.4f} | {row['tv_used']:8.4f} | "
            f"{row['raw_scale']:10.4f} | {row['adaptive_scale']:11.4f} | {row['rew_scale']:10.4f}"
        )


def print_foot_clearance_table(term_name, step_heights, step_width, clip_value, num_offsets, std_gain, max_extra_clearance, stance_gain, swing_high_penalty_weight):
    x_points, _ = make_under_body_grid()

    print(f"\nCurrent {term_name} adaptive model")
    print(TERM_CONFIGS[term_name]["formula"])
    print("stance_tolerance = 0.02 + stance_gain * extra_clearance")
    print("swing high-penalty starts above swing_target + extra_clearance")
    print(
        "terrain_variability is computed from under-body height std over "
        f"{len(UNDER_BODY_POINTS_X)} x {len(UNDER_BODY_POINTS_Y)} samples"
    )
    print(
        f"step_width = {step_width:.3f} m, clip = {clip_value:.3f}, std_gain = {std_gain:.3f}, "
        f"max_extra_clearance = {max_extra_clearance:.3f}, stance_gain = {stance_gain:.3f}, "
        f"swing_high_penalty_weight = {swing_high_penalty_weight:.3f}"
    )

    header = (
        f"{'step_h':>8} | {'std_max':>8} | {'extra_max':>10} | {'stance_max':>10} | "
        f"{'std_mean':>8} | {'extra_mean':>10} | {'stance_mean':>11} | {'offset@max':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for step_height in step_heights:
        rows = evaluate_foot_clearance_case(
            x_points,
            step_width,
            step_height,
            clip_value,
            std_gain,
            max_extra_clearance,
            stance_gain,
            swing_high_penalty_weight,
            num_offsets,
        )
        most_relaxed, _, mean_std, mean_extra_clearance, mean_stance_tolerance = summarize_foot_clearance_rows(rows)
        print(
            f"{step_height:8.3f} | {most_relaxed['std']:8.4f} | {most_relaxed['extra_clearance']:10.4f} | "
            f"{most_relaxed['stance_tolerance']:10.4f} | {mean_std:8.4f} | {mean_extra_clearance:10.4f} | "
            f"{mean_stance_tolerance:11.4f} | {most_relaxed['offset']:10.4f}"
        )


def print_foot_clearance_detail(term_name, step_height, step_width, clip_value, num_offsets, std_gain, max_extra_clearance, stance_gain, swing_high_penalty_weight):
    x_points, _ = make_under_body_grid()
    rows = evaluate_foot_clearance_case(
        x_points,
        step_width,
        step_height,
        clip_value,
        std_gain,
        max_extra_clearance,
        stance_gain,
        swing_high_penalty_weight,
        num_offsets,
    )
    most_relaxed, least_relaxed, mean_std, mean_extra_clearance, mean_stance_tolerance = summarize_foot_clearance_rows(rows)

    print(f"\nDetailed {term_name} scan for step height = {step_height:.3f} m")
    print(
        f"mean std = {mean_std:.4f}, mean extra_clearance = {mean_extra_clearance:.4f}, "
        f"mean stance_tolerance = {mean_stance_tolerance:.4f}, "
        f"swing_high_penalty_weight = {swing_high_penalty_weight:.4f}"
    )
    print(
        "most relaxed offset: "
        f"offset = {most_relaxed['offset']:.4f}, std = {most_relaxed['std']:.4f}, "
        f"tv_used = {most_relaxed['tv_used']:.4f}, raw_extra_clearance = {most_relaxed['raw_extra_clearance']:.4f}, "
        f"extra_clearance = {most_relaxed['extra_clearance']:.4f}, stance_tolerance = {most_relaxed['stance_tolerance']:.4f}"
    )
    print(
        "least relaxed offset: "
        f"offset = {least_relaxed['offset']:.4f}, std = {least_relaxed['std']:.4f}, "
        f"tv_used = {least_relaxed['tv_used']:.4f}, raw_extra_clearance = {least_relaxed['raw_extra_clearance']:.4f}, "
        f"extra_clearance = {least_relaxed['extra_clearance']:.4f}, stance_tolerance = {least_relaxed['stance_tolerance']:.4f}"
    )

    print("\noffset scan")
    print(
        f"{'offset':>8} | {'std':>8} | {'tv_used':>8} | {'raw_extra':>10} | "
        f"{'extra':>8} | {'stance_tol':>10}"
    )
    print("-" * 68)
    for row in rows:
        print(
            f"{row['offset']:8.4f} | {row['std']:8.4f} | {row['tv_used']:8.4f} | "
            f"{row['raw_extra_clearance']:10.4f} | {row['extra_clearance']:8.4f} | {row['stance_tolerance']:10.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate terrain-adaptive reward scaling for stairs using the current black_env algorithm."
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        choices=sorted(TERM_CONFIGS.keys()),
        default=["orientation", "action_rate", "smoothness", "foot_clearance"],
        help="Reward terms to print.",
    )
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    parser.add_argument("--step-width", type=float, default=DEFAULT_STEP_WIDTH)
    parser.add_argument(
        "--step-heights",
        type=float,
        nargs="+",
        default=DEFAULT_STEP_HEIGHTS,
    )
    parser.add_argument(
        "--detail-step-height",
        type=float,
        default=None,
        help="If set, also print a full offset scan for this stair height.",
    )
    parser.add_argument(
        "--num-offsets",
        type=int,
        default=DEFAULT_NUM_OFFSETS,
        help="Number of offsets to scan inside one stair period.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for term_name in args.terms:
        cfg = TERM_CONFIGS[term_name]
        if cfg["kind"] == "decay":
            print_decay_term_table(
                term_name=term_name,
                step_heights=args.step_heights,
                step_width=args.step_width,
                clip_value=args.clip,
                num_offsets=args.num_offsets,
                sigma=cfg["sigma"],
                min_scale=cfg["min_scale"],
                max_scale=cfg["max_scale"],
                original_scale=cfg["original_scale"],
            )
            if args.detail_step_height is not None:
                print_decay_term_detail(
                    term_name=term_name,
                    step_height=args.detail_step_height,
                    step_width=args.step_width,
                    clip_value=args.clip,
                    num_offsets=args.num_offsets,
                    sigma=cfg["sigma"],
                    min_scale=cfg["min_scale"],
                    max_scale=cfg["max_scale"],
                    original_scale=cfg["original_scale"],
                )
        elif cfg["kind"] == "margin":
            print_foot_clearance_table(
                term_name=term_name,
                step_heights=args.step_heights,
                step_width=args.step_width,
                clip_value=args.clip,
                num_offsets=args.num_offsets,
                std_gain=cfg["std_gain"],
                max_extra_clearance=cfg["max_extra_clearance"],
                stance_gain=cfg["stance_gain"],
                swing_high_penalty_weight=cfg["swing_high_penalty_weight"],
            )
            if args.detail_step_height is not None:
                print_foot_clearance_detail(
                    term_name=term_name,
                    step_height=args.detail_step_height,
                    step_width=args.step_width,
                    clip_value=args.clip,
                    num_offsets=args.num_offsets,
                    std_gain=cfg["std_gain"],
                    max_extra_clearance=cfg["max_extra_clearance"],
                    stance_gain=cfg["stance_gain"],
                    swing_high_penalty_weight=cfg["swing_high_penalty_weight"],
                )
