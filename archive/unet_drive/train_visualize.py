import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def plot_loss(exp_folder, output_image_path="loss_plot.png", title="Loss Plot"):
    # フォルダ名に基づいてtrain_logとval_logのパスを設定
    train_log_path = os.path.join(exp_folder, "train_log.csv")
    val_log_path = os.path.join(exp_folder, "val_log.csv")
    
    # CSVファイルの読み込み
    train_log = pd.read_csv(train_log_path)
    val_log = pd.read_csv(val_log_path)
    
    # 損失をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(train_log['Epoch'], train_log['Loss'], label='Train Loss')
    plt.plot(val_log['Epoch'], val_log['Loss'], label='Validation Loss', linestyle='--')
    
    # グラフの装飾
    plt.title(title)  # ここでタイトルを設定
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # グラフを保存
    plt.savefig(output_image_path)
    plt.show()

if __name__ == "__main__":
    # argparseを使ってコマンドライン引数を受け取る
    parser = argparse.ArgumentParser(description='Plot training and validation loss.')
    parser.add_argument('save_dir', type=str, help='Directory where experiment logs are saved')
    
    # 引数を解析
    args = parser.parse_args()
    
    # save_dirに基づいてログのパスを設定
    exp_folder = os.path.join("exp", args.save_dir)
    output_image_path = os.path.join("result", f"loss_plot_{args.save_dir}.png")
    
    # プロットの実行 (タイトルに args.save_dir を使用)
    plot_loss(exp_folder, output_image_path, title=args.save_dir)
