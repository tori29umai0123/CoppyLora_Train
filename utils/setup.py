import os
import sys
import subprocess

def is_colab_environment():
    """Google Colab環境かどうかをチェックする関数"""
    return 'google.colab' in sys.modules

def run_command(command, cwd=None):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        print(f"Command succeeded: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr}")

def install_dependencies_and_setup():
    if is_colab_environment():
        base_dir = '/content/CoppyLora_Train'
        print("Running in Google Colab, changed directory to /content/CoppyLora_Train")

    # sd-scripts ディレクトリのパスを指定
    sd_scripts_dir = os.path.join(base_dir, 'sd-scripts') if is_colab_environment() else 'sd-scripts'

    # sd-scripts をクローン
    run_command(f"git clone -b dev https://github.com/kohya-ss/sd-scripts.git {sd_scripts_dir}")

    # その他の依存関係のインストールコマンド
    commands = [
        "pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118",
        "pip install -U xformers",
        "pip install --upgrade -r requirements.txt",  # sd-scripts ディレクトリの requirements.txt を指定
        "pip install bitsandbytes==0.41.1",
        "pip install scipy",
        "pip install wget",
    ]

    # sd-scripts ディレクトリでコマンドを実行
    for cmd in commands:
        run_command(cmd, cwd=sd_scripts_dir)

# スクリプトの実行
install_dependencies_and_setup()
