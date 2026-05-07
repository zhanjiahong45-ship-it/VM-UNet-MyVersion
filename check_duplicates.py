"""
医学影像数据集双重查重工具
==================================
用于对比你刚刚筛选出的“测试集”与你现有的“训练集/其他数据集”。
防止数据泄露 (Data Leakage)。

双重检测机制：
  1. 文件名匹配：直接抓取同名文件 (如 ISIC_0012345)
  2. MD5 像素哈希匹配：即使文件被重命名，只要像素一致也能查出

用法：
  python check_duplicates.py --source_dir <刚才筛选出的文件夹> --target_dir <你现有的其他数据集文件夹>
"""

import os
import hashlib
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def get_md5(file_path: str) -> str:
    """计算文件的 MD5 哈希值（读取大文件也不会爆内存）"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        return f"ERROR_{e}"


def scan_directory(directory: Path, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif")):
    """扫描目录，返回文件名映射和哈希映射"""
    files = [f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in extensions]

    file_map = {}  # {文件名 (不含后缀): 完整路径}
    hash_map = {}  # {MD5哈希值: 完整路径}

    print(f"正在扫描并计算哈希: {directory} (共 {len(files)} 个文件)...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        # 并发计算 MD5 加速扫描
        hash_results = list(executor.map(get_md5, [str(f) for f in files]))

    for f, md5_val in zip(files, hash_results):
        file_map[f.stem] = f
        if not md5_val.startswith("ERROR"):
            hash_map[md5_val] = f

    return file_map, hash_map


def check_duplicates(source_dir: str, target_dir: str, output_txt: str):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists() or not target_dir.exists():
        print("[ERROR] 目录不存在，请检查路径。")
        return

    print("=" * 50)
    print("开始数据集查重...")
    print("=" * 50)

    # 1. 扫描目标目录（你的旧数据集库）
    target_file_map, target_hash_map = scan_directory(target_dir)

    # 2. 扫描源目录（你刚才筛选出的 100 多张新图）
    source_file_map, source_hash_map = scan_directory(source_dir)

    duplicates_by_name = []
    duplicates_by_hash = []

    # 3. 对比文件名 (忽略后缀，比如 .jpg 和 .png 同名也算)
    for stem, src_path in source_file_map.items():
        if stem in target_file_map:
            duplicates_by_name.append((src_path, target_file_map[stem]))

    # 4. 对比像素哈希 (针对改名的情况)
    for md5_val, src_path in source_hash_map.items():
        if md5_val in target_hash_map:
            # 如果名字不一样，但哈希一样，说明是改名文件
            tgt_path = target_hash_map[md5_val]
            if src_path.stem != tgt_path.stem:
                duplicates_by_hash.append((src_path, tgt_path))

    # 5. 输出报告
    print("\n" + "=" * 50)
    print("查重结果报告")
    print("=" * 50)

    total_dups = len(duplicates_by_name) + len(duplicates_by_hash)

    if total_dups == 0:
        print("🎉 恭喜！没有发现任何重复的数据。这批测试集非常干净！")
    else:
        print(f"⚠️ 警告：发现了 {total_dups} 张重复图片！\n")

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("数据集查重报告\n")
            f.write("=" * 30 + "\n")

            if duplicates_by_name:
                print(f"[{len(duplicates_by_name)} 张] 文件名冲突 (可能在两个集合中都存在):")
                f.write(f"\n一、文件名冲突 ({len(duplicates_by_name)}张):\n")
                for src, tgt in duplicates_by_name:
                    msg = f"  新图: {src.name}  <==重复==>  旧图库: {tgt}"
                    print(msg)
                    f.write(msg + "\n")

            if duplicates_by_hash:
                print(f"\n[{len(duplicates_by_hash)} 张] 内容哈希冲突 (名字不同但图片完全一样):")
                f.write(f"\n二、哈希冲突-图片完全一致 ({len(duplicates_by_hash)}张):\n")
                for src, tgt in duplicates_by_hash:
                    msg = f"  新图: {src.name}  <==重复==>  旧图库: {tgt}"
                    print(msg)
                    f.write(msg + "\n")

        print(f"\n[OUTPUT] 详细冲突名单已保存至: {output_txt}")
        print("建议：请将这些冲突图片从你的'测试集'中移除，以防数据泄露。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双重查重脚本")
    parser.add_argument("--source", required=True, help="刚刚筛选出来的 100 多张图的目录")
    parser.add_argument("--target", required=True, help="你要对比的现存大本营目录（比如你的训练集）")
    parser.add_argument("--output", default="duplicate_report.txt", help="报告输出路径")

    args = parser.parse_args()
    check_duplicates(args.source, args.target, args.output)