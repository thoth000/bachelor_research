#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from PIL import Image, ImageDraw

def draw_rectangle(draw_obj, box, color, width=3):
    """
    box: (left, top, right, bottom) のタプルで指定された範囲に枠線を描画する。
    color: 枠線の色（例: "red", "green", (R, G, B) 等）。
    width: 枠線の太さ。
    """
    for i in range(width):
        draw_obj.rectangle(
            [box[0] + i, box[1] + i, box[2] - i, box[3] - i],
            outline=color
        )

def process_image(input_path, output_dir,
                  red_crop_box=(10, 10, 74, 74),       # (left, top, right, bottom)
                  green_crop_box=(100, 100, 164, 164), # (left, top, right, bottom)
                  outline_width=3,
                  final_height=592):
    """
    1. 画像を読み込み
    2. 指定の赤領域・緑領域を元画像に枠線付きで描画
    3. 赤領域・緑領域をクロップ & 各クロップ画像にも枠線を付与
    4. クロップ画像をリサイズ (ここでは、高さの合計を元画像と同じにする想定)
    5. 元画像の右側に縦に並べて貼り付け
    6. output_dir に "元画像名_connected.拡張子" という形式で保存
    """
    img = Image.open(input_path).convert("RGB")
    w, h = img.size  # 万が一 592x592 でなくても動くように

    # 元画像に枠線を描く
    draw_img = ImageDraw.Draw(img)
    draw_rectangle(draw_img, red_crop_box, color="orange", width=outline_width)
    draw_rectangle(draw_img, green_crop_box, color="blue", width=outline_width)

    # クロップして別画像として取り出す (赤)
    red_crop = img.crop(red_crop_box)
    draw_red = ImageDraw.Draw(red_crop)
    rw, rh = red_crop.size
    draw_rectangle(draw_red, (0, 0, rw, rh), color="orange", width=outline_width)

    # クロップして別画像として取り出す (緑)
    green_crop = img.crop(green_crop_box)
    draw_green = ImageDraw.Draw(green_crop)
    gw, gh = green_crop.size
    draw_rectangle(draw_green, (0, 0, gw, gh), color="blue", width=outline_width)

    # リサイズ
    # ここでは、クロップ画像2枚を上下に並べて高さが元画像と同じになるように分割
    half_height = final_height // 2  # 赤と緑で高さを2分割
    red_resized = red_crop.resize((half_height, half_height))
    green_resized = green_crop.resize((half_height, half_height))

    # 結合先の画像を作成
    out_w = w + half_height  # 元画像幅 + リサイズ後クロップの幅
    out_h = h                # 元画像と同じ高さ
    new_img = Image.new("RGB", (out_w, out_h), (255, 255, 255))

    # 左に元画像、右上に赤、右下に緑を配置
    new_img.paste(img, (0, 0))
    new_img.paste(red_resized, (w, 0))
    new_img.paste(green_resized, (w, half_height))

    # 保存用パスを整形
    basename = os.path.basename(input_path)
    root, ext = os.path.splitext(basename)
    out_name = f"{root}_connected{ext}"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)
    new_img.save(out_path)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="592x592 画像から 2 箇所の領域をクロップし結合するスクリプト"
    )
    parser.add_argument("input_dir", type=str,
                        help="処理対象の画像が置かれたディレクトリのパス")
    parser.add_argument("--output_dir", type=str, default="connected",
                        help="処理結果を保存するディレクトリ（デフォルト: connected）")
    parser.add_argument("--size", type=int, default=64)
    # 必要に応じて座標などを引数化したい場合はオプションを増やしてください
    args = parser.parse_args()

    # 画像ディレクトリと出力ディレクトリ
    input_dir = args.input_dir
    output_dir = args.output_dir

    # 処理対象とする拡張子
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    red_tl = (460, 255)  # 赤の左上座標
    green_tl = (200, 300)  # 緑の左上座標

    # クロップ座標（必要に応じて調整してください）
    red_box = (red_tl[0], red_tl[1], red_tl[0]+args.size, red_tl[1]+args.size)
    green_box = (green_tl[0], green_tl[1], green_tl[0]+args.size, green_tl[1]+args.size)

    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        # 拡張子で画像かどうかを判定
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in valid_exts:
            process_image(
                input_path=fpath,
                output_dir=output_dir,
                red_crop_box=red_box,
                green_crop_box=green_box,
                outline_width=3,
                final_height=592
            )


if __name__ == "__main__":
    main()
