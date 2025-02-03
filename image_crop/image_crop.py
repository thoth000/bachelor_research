#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from PIL import Image

def process_image(input_path, output_dir, blue_crop_box):
    """
    【処理内容】
      1. 画像を読み込み
      2. 指定の青領域(正方形)をクロップ
      3. output_dir に「元ファイル名_bluecrop.拡張子」という形式で保存
    """
    # 画像を開く
    img = Image.open(input_path).convert("RGB")

    # 青領域をクロップ
    blue_crop = img.crop(blue_crop_box)

    # 出力用ファイル名を組み立て
    basename = os.path.basename(input_path)
    root, ext = os.path.splitext(basename)
    out_name = f"{root}_bluecrop{ext}"

    # 出力先ディレクトリを作成(存在しない場合)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)

    # クロップ画像を保存
    blue_crop.save(out_path)
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="1箇所の青領域のみをクロップして出力するスクリプト (枠線なし)"
    )
    parser.add_argument("input_dir", type=str,
                        help="処理対象の画像が置かれたディレクトリのパス")
    parser.add_argument("--output_dir", type=str, default="connected",
                        help="処理結果を保存するディレクトリ（デフォルト: connected）")
    parser.add_argument("--size", type=int, default=64,
                        help="クロップする正方形の一辺サイズ（デフォルト: 64）")
    args = parser.parse_args()

    # パラメータ取得
    input_dir = args.input_dir
    output_dir = args.output_dir
    crop_size = args.size

    # クロップしたい領域の左上座標を指定 (例: (200, 300))
    # 必要に応じて変更してください
    blue_tl = (200, 300)
    # blue_tl = (270, 315)
    # クロップ領域 (left, top, right, bottom) を計算
    blue_box = (
        blue_tl[0],
        blue_tl[1],
        blue_tl[0] + crop_size,
        blue_tl[1] + crop_size
    )

    # 対象とする画像拡張子
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # 指定ディレクトリ内のファイルを走査
    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        # 拡張子で画像かどうか判定
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in valid_exts:
            process_image(
                input_path=fpath,
                output_dir=output_dir,
                blue_crop_box=blue_box
            )

if __name__ == "__main__":
    main()
