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
DEFAULT_SIGMA = 0.009
DEFAULT_MIN_SCALE = 0.10
DEFAULT_MAX_SCALE = 1.00
DEFAULT_ORIGINAL_SCALE = -3.0


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


def evaluate_stair_case(
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
        tv_used, raw_scale, final_scale = adaptive_scale(
            std, sigma, clip_value, min_scale, max_scale
        )
        rows.append(
            {
                "offset": float(offset),
                "std": std,
                "tv_used": tv_used,
                "raw_scale": raw_scale,
                "pitch_scale": final_scale,
                "rew_scale": original_scale * final_scale,
            }
        )
    return rows


def summarize_rows(rows):
    best_relaxed = min(rows, key=lambda row: row["pitch_scale"])
    least_relaxed = max(rows, key=lambda row: row["pitch_scale"])
    mean_pitch_scale = float(np.mean([row["pitch_scale"] for row in rows]))
    mean_rew_scale = float(np.mean([row["rew_scale"] for row in rows]))
    mean_std = float(np.mean([row["std"] for row in rows]))
    return best_relaxed, least_relaxed, mean_pitch_scale, mean_rew_scale, mean_std


def print_stair_table(
    step_heights,
    step_width,
    sigma,
    clip_value,
    min_scale,
    max_scale,
    original_scale,
    num_offsets,
):
    x_points, _ = make_under_body_grid()

    print("\nCurrent orientation adaptive model")
    print("pitch_scale = clamp(exp(-(terrain_variability^2) / sigma), min_scale, max_scale)")
    print("final pitch reward scale = original_orientation_scale * pitch_scale")
    print(
        "terrain_variability is computed from under-body height std over "
        f"{len(UNDER_BODY_POINTS_X)} x {len(UNDER_BODY_POINTS_Y)} samples"
    )
    print(
        f"step_width = {step_width:.3f} m, clip = {clip_value:.3f}, sigma = {sigma:.4f}, "
        f"min_scale = {min_scale:.3f}, max_scale = {max_scale:.3f}"
    )
    print(f"original orientation scale = {original_scale:.3f}")

    header = (
        f"{'step_h':>8} | {'std_max':>8} | {'pitch_min':>9} | {'rew_min':>9} | "
        f"{'std_mean':>8} | {'pitch_mean':>10} | {'rew_mean':>9} | {'offset@min':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for step_height in step_heights:
        rows = evaluate_stair_case(
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
        best_relaxed, _, mean_pitch_scale, mean_rew_scale, mean_std = summarize_rows(rows)
        print(
            f"{step_height:8.3f} | {best_relaxed['std']:8.4f} | {best_relaxed['pitch_scale']:9.4f} | "
            f"{best_relaxed['rew_scale']:9.4f} | {mean_std:8.4f} | {mean_pitch_scale:10.4f} | "
            f"{mean_rew_scale:9.4f} | {best_relaxed['offset']:10.4f}"
        )


def print_single_height_detail(
    step_height,
    step_width,
    sigma,
    clip_value,
    min_scale,
    max_scale,
    original_scale,
    num_offsets,
):
    x_points, _ = make_under_body_grid()
    rows = evaluate_stair_case(
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
    best_relaxed, least_relaxed, mean_pitch_scale, mean_rew_scale, mean_std = summarize_rows(rows)

    print(f"\nDetailed stair scan for step height = {step_height:.3f} m")
    print(
        f"mean std = {mean_std:.4f}, mean pitch_scale = {mean_pitch_scale:.4f}, "
        f"mean rew_scale = {mean_rew_scale:.4f}"
    )
    print(
        "most relaxed offset: "
        f"offset = {best_relaxed['offset']:.4f}, std = {best_relaxed['std']:.4f}, "
        f"tv_used = {best_relaxed['tv_used']:.4f}, raw_scale = {best_relaxed['raw_scale']:.4f}, "
        f"pitch_scale = {best_relaxed['pitch_scale']:.4f}, rew_scale = {best_relaxed['rew_scale']:.4f}"
    )
    print(
        "least relaxed offset: "
        f"offset = {least_relaxed['offset']:.4f}, std = {least_relaxed['std']:.4f}, "
        f"tv_used = {least_relaxed['tv_used']:.4f}, raw_scale = {least_relaxed['raw_scale']:.4f}, "
        f"pitch_scale = {least_relaxed['pitch_scale']:.4f}, rew_scale = {least_relaxed['rew_scale']:.4f}"
    )

    print("\noffset scan")
    print(
        f"{'offset':>8} | {'std':>8} | {'tv_used':>8} | {'raw_scale':>10} | "
        f"{'pitch_scale':>11} | {'rew_scale':>10}"
    )
    print("-" * 74)
    for row in rows:
        print(
            f"{row['offset']:8.4f} | {row['std']:8.4f} | {row['tv_used']:8.4f} | "
            f"{row['raw_scale']:10.4f} | {row['pitch_scale']:11.4f} | {row['rew_scale']:10.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate terrain-adaptive orientation scaling for stairs using the current black_env algorithm."
    )
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    parser.add_argument("--min-scale", type=float, default=DEFAULT_MIN_SCALE)
    parser.add_argument("--max-scale", type=float, default=DEFAULT_MAX_SCALE)
    parser.add_argument("--original-scale", type=float, default=DEFAULT_ORIGINAL_SCALE)
    parser.add_argument("--step-width", type=float, default=DEFAULT_STEP_WIDTH)
    parser.add_argument(
        "--step-heights",
        type=float,
        nargs="+",
        default=[0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.23],
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
        default=30,
        help="Number of offsets to scan inside one stair period.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_stair_table(
        step_heights=args.step_heights,
        step_width=args.step_width,
        sigma=args.sigma,
        clip_value=args.clip,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        original_scale=args.original_scale,
        num_offsets=args.num_offsets,
    )
    if args.detail_step_height is not None:
        print_single_height_detail(
            step_height=args.detail_step_height,
            step_width=args.step_width,
            sigma=args.sigma,
            clip_value=args.clip,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            original_scale=args.original_scale,
            num_offsets=args.num_offsets,
        )
