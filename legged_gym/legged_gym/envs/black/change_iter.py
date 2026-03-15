import argparse
import os
import re
import shutil

import torch


def infer_iter_from_filename(path: str) -> int:
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
    if not match:
        raise ValueError(f"无法从文件名推断迭代号: {path}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description="修复 checkpoint 中记录的 iter 字段")
    parser.add_argument("path", help="checkpoint 路径，例如 model_3820.pt")
    parser.add_argument(
        "--iter",
        dest="target_iter",
        type=int,
        default=None,
        help="目标 iter；不传则默认从文件名 model_XXXX.pt 自动推断",
    )
    parser.add_argument(
        "--save-backup",
        action="store_true",
        help="自动生成 .bak 备份文件",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到 checkpoint 文件: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    old_iter = checkpoint.get("iter", None)
    target_iter = args.target_iter if args.target_iter is not None else infer_iter_from_filename(checkpoint_path)

    print(f"文件: {checkpoint_path}")
    print(f"原 iter: {old_iter}")
    print(f"目标 iter: {target_iter}")

    if  args.save_backup:
        backup_path = checkpoint_path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(checkpoint_path, backup_path)
            print(f"已创建备份: {backup_path}")
        else:
            print(f"备份已存在，跳过创建: {backup_path}")

    checkpoint["iter"] = target_iter
    torch.save(checkpoint, checkpoint_path)
    print("修复完成")


if __name__ == "__main__":
    main()
