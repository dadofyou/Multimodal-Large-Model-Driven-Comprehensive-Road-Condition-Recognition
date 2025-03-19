import requests
import os

# 模型下载链接
model_urls = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    # "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    # "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

# 目标目录
save_dir = "models"

# 确保目录存在
os.makedirs(save_dir, exist_ok=True)

# 下载模型
for model_name, url in model_urls.items():
    file_path = os.path.join(save_dir, f"{model_name}.pth")
    print(f"Downloading {model_name} to {file_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{model_name} downloaded successfully.")
    else:
        print(f"Failed to download {model_name}. Status code: {response.status_code}")