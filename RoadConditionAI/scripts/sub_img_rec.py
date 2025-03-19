import logging
import cv2
import torch
import clip
import numpy as np
from typing import List, Dict, Callable
from RoadConditionAI.configs import settings
import torch
from PIL import Image

"""
clip 环境配置
激活环境配置
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
"""

# 初始化 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

# 提取所有子类型文本标签
subtypes = []
for parent, children in settings.SUBTYPE_MAPPING.items():
    subtypes.extend(children)

# 将子类型转换为CLIP可处理的文本
text_inputs = [f"a photo of {t}" for t in subtypes]
text_tokens = clip.tokenize(text_inputs).to(device)

with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)  # 标准化


def classify_with_clip(
        cropped_image: np.ndarray,
        subtype_mapping: Dict[str, List[str]],
        model: torch.nn.Module = clip_model,
        preprocess: Callable = clip_preprocess,
        text_features: torch.Tensor = text_features,
        subtypes: List[str] = subtypes
) -> Dict[str, str]:
    """
    使用CLIP对裁剪后的图像进行分类，返回父类和子类标签。

    Args:
        cropped_image: 裁剪后的图像（BGR格式，OpenCV格式）
        model: CLIP模型
        preprocess: 图像预处理函数
        text_features: 预先计算的文本特征
        subtypes: 所有可能的子类型列表
        subtype_mapping: 类别映射字典（如SUBTYPE_MAPPING）

    Returns:
        包含parent_label和subtype的字典，或空字典（分类失败）
    """
    try:
        # 图像预处理
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 计算相似度并选择最佳子类型
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            best_subtype = subtypes[indices[0].item()]

            # 映射到父类别
            parent_label = None
            for parent, children in subtype_mapping.items():
                if best_subtype in children:
                    parent_label = parent
                    break

            return {
                "parent_label": parent_label,
                "subtype": best_subtype
            }
    except Exception as e:
        logging.error(f"CLIP分类失败: {str(e)}")
        return {}
