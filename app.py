import gradio as gr
import torch
import subprocess
import os
from PIL import Image
import os

path = os.getcwd()
SDXL_dir = os.path.join(path, "SDXL")
os.makedirs(SDXL_dir, exist_ok=True)
SDXL_model = os.path.join(SDXL_dir, "animagine-xl-3.1.safetensors")
lora_dir = os.path.join(path, "lora")
os.makedirs(lora_dir, exist_ok=True)
base_lora = os.path.join(lora_dir, "copi-ki-base-c.safetensors")
base_image_dir = os.path.join(path, "base_image")
os.makedirs(base_image_dir, exist_ok=True)
base_image_path = os.path.join(base_image_dir, "base_c_1024.png")
output_dir = os.path.join(path, "output")
os.makedirs(output_dir, exist_ok=True)
accelerate_config = os.path.join(path, "accelerate_config.yaml")
path_to_train_util = os.path.join(path,'sd-scripts/library/train_util.py')
path_to_sdxl_train_network = os.path.join(path,'sd-scripts/sdxl_train_network.py')
SDXL_config = os.path.join(path, "copi-ki_SDXL.toml")

def replace_text_in_file(file_path, search_text, replace_text):
    # ファイルを読み込む
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # テキストを置換する
    data = data.replace(search_text, replace_text)
    
    # ファイルに書き込む
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)

replace_text_in_file(path_to_train_util, 'config_file', 'train_config_file')
replace_text_in_file(path_to_sdxl_train_network, 'config_file', 'train_config_file')


def train(input_image_path, lora_name, mode_inputs):
    input_image = Image.open(input_image_path)
    if mode_inputs == "Lineart":
        lineart_dir = os.path.join(path, "lineart/4000")
        train_dir = os.path.join(path, "lineart")
        for size in [1024, 768, 512]:
            resize_image = input_image.resize((size, size))
            resize_image.save(os.path.join(lineart_dir, f"{size}.png"))

        command1 = [
            "accelerate", "launch", "--config_file", accelerate_config, "sdxl_train_network.py",
            "--pretrained_model_name_or_path", SDXL_model,
            "--train_data_dir", train_dir,
            "--train_config_file", SDXL_config,
            "--output_dir", lora_dir,
        ]
        subprocess.run(command1, check=True, cwd=os.path.join(path, "sd-scripts"))

        trained_lora = os.path.join(lora_dir, "copi-ki-kari.safetensors")

        command2 = [
            "python", "networks/sdxl_merge_lora.py",
            "--save_to", os.path.join(lora_dir, f"{lora_name}.safetensors"),
            "--models", trained_lora, base_lora,
            "--ratios", "0.7", "0.6", "1.0",
            "--new_rank", "16",
            "--device", "cuda",
            "--save_precision", "bf16"
        ]
        subprocess.run(command2, check=True, cwd=os.path.join(path, "sd-scripts"))

        return os.path.join(lora_dir, f"{lora_name}.safetensors")

def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                base_img = gr.Image(value=base_image_path, label="Base Image")
                input_image_path = gr.Image(label="Input Image", type='filepath')
                lora_name = gr.Textbox(label="LoRa Name", value="mylora")
                mode_inputs = gr.Dropdown(label="Mode", choices=["Lineart"], value="Lineart")
                train_button = gr.Button("Train")
            with gr.Column():
                output_file = gr.File(label="Download Output File")

        train_button.click(
            fn=train,
            inputs=[input_image_path, lora_name, mode_inputs],
            outputs=output_file
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()
