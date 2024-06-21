import os

path = os.getcwd()
path_to_train_util = os.path.join(path,'sd-scripts/library/train_util.py')

def replace_text_in_file_if_needed(file_path, search_text, replace_text):
    # ファイルを読み込む
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # すでに置換されている場合は何もしない
    if replace_text in data:
        print(f"No need to replace text in {file_path}, already replaced.")
        return

    # テキストが存在していれば置換する
    if search_text in data:
        # テキストを置換する
        updated_data = data.replace(search_text, replace_text)
        
        # ファイルに書き込む
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_data)
        print(f"Replaced text in {file_path}")
    else:
        print(f"{search_text} not found in {file_path}. No replacement necessary.")

replace_text_in_file_if_needed(path_to_train_util, 'config_file', 'train_config_file')
