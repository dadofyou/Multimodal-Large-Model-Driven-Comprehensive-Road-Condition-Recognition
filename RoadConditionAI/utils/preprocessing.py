# utils/preprocessing.py

import cv2


def preprocess_image(image_path, target_size=(1208, 1208)):
    """
    对图片进行预处理，例如调整分辨率。
    返回处理后的图片路径（可覆盖原文件或保存到临时目录）。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    processed_img = cv2.resize(img, target_size)
    # 可以选择保存到原路径或其他路径
    cv2.imwrite(image_path, processed_img)
    return image_path
