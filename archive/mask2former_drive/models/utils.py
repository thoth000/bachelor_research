def read_mask2former_token(token_file_path):
    try:
        with open(token_file_path, 'r') as file:
            content = file.read().strip()  # 改行や空白を取り除く
        return content
    except FileNotFoundError:
        print("ファイルが見つかりませんでした。")
        return None
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None
