import os

import requests
from tqdm import tqdm


def dl_SDXL_model(model_dir):
    file_name = 'animagine-xl-3.1.safetensors'
    file_path = os.path.join(model_dir, file_name)
    if not os.path.exists(file_path):
        url = "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors"
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded {file_name}')
        else:
            print(f'Failed to download {file_name}')
    else:
        print(f'{file_name} already exists.')




def dl_lora_model(model_dir):
    file_name = 'copi-ki-base-c.safetensors'
    file_path = os.path.join(model_dir, file_name)
    if not os.path.exists(file_path):
        url = "https://huggingface.co/tori29umai/mylora/resolve/main/copi-ki-base-c.safetensors"
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded {file_name}')
        else:
            print(f'Failed to download {file_name}')
    else:
        print(f'{file_name} already exists.')


def dl_base_image(image_dir):
    file_name = 'base_c_1024.png'
    file_path = os.path.join(image_dir, file_name)
    if not os.path.exists(file_path):
        url = "https://huggingface.co/tori29umai/mylora/resolve/main/base_c_1024.png?download=true"
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded {file_name}')
        else:
            print(f'Failed to download {file_name}')
    else:
        print(f'{file_name} already exists.')

