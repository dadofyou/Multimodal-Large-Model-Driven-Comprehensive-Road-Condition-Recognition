import logging
from pathlib import Path

import torch
import json
import clip
import numpy as np
import torch
from PIL import Image
from RoadConditionAI.configs.settings import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_VIS_ANN_DIR,
    DATA_PROCESSED_VIS_ANN_IMG_DIR
)

# 确保输出目录存在
DATA_PROCESSED_VIS_ANN_DIR.mkdir(parents=True, exist_ok=True)

"""
clip 环境配置
激活环境配置
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
"""

# 初始化 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
"""
CLIP筛选目标
除去无意义的分割区块（由于SAM分割结果较细，有很多无意义的分割结果。）
筛选出：车辆，道路等个体，以便违规占道/交通事故的判断。
筛选出：抛洒物，交通标志线，路面裂缝等物体或现象，以便使用模型对其进行进一步识别。
"""
# 定义道路检测语义库（根据实际需求补充完整）
ROAD_STATUS_CONFIG = {
    "text_prompts": {
        # 使用具体车辆类型替代抽象概念
        "car": [
            # 车辆独有特征
            "A motor vehicle with four wheels and visible windows",
            "Road vehicle with tires and headlights",
            "Automobile having windshield and license plate",

            # 添加否定约束
            "A car not including trees or poles",
            "Vehicle excluding vertical structures like columns",

            # 多视角描述
            "Side view of sedan car with visible wheels",
            "Rear perspective of pickup truck with tailgate",

            # 物理接触特征
            "Car tires in contact with road surface",
            "Vehicle casting shadow on asphalt"
        ],
        "crack": [
            # 裂缝类型学强化
            "Asphalt alligator cracking with 5-15cm polygon patterns",  # 龟裂
            "Transverse crack perpendicular to road axis (spacing>2m)",  # 横向裂缝
      ],
        "pothole": [
            "Excluding drainage grates through pattern regularity analysis",
            "Not confused with speed bumps via longitudinal profile scanning"
        ],
        "lines": [
            # **共性特征（所有线条类物体）**
            "A white rectangle or parallelogram",
            "painted lane line",

            "White painted line on asphalt surface with sharp edges",
            "Straight line with angle 0-180° relative to road direction",
        ],
        "unknown": [
            # 其他可能干扰物
            "Overhead traffic signs on gantries",

            # 特殊天气干扰
            "Rainwater reflections mimicking line markings",
            "Leaf debris arranged in linear patterns",
            "unknown",
            "items with such as trees and big pillar."
        ]
    },
    "category_weights": {
        "car": 0.26,
        "crack": 0.25,
        "pothole": 0.25,
        "lines": 0.25,
        "unknown": 0.25,
    }
}


# 预计算文本特征（带类别权重）
text_features_dict = {}
# 修改文本特征处理部分（预计算部分）
for category, prompts in ROAD_STATUS_CONFIG["text_prompts"].items():
    text_inputs = torch.cat([clip.tokenize(f"a photo of {desc}") for desc in prompts]).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(text_inputs).float()
        # 先归一化再应用权重
        features /= features.norm(dim=-1, keepdim=True)
        weighted_features = features * ROAD_STATUS_CONFIG["category_weights"][category]
    text_features_dict[category] = weighted_features


def filter_with_clip(raw_img_path: Path):
    """处理单张图片的所有分割区域"""
    # 获取对应的分割JSON文件
    global similarities
    json_path = DATA_PROCESSED_DIR / f"{raw_img_path.stem}_sub.json"
    if not json_path.exists():
        logging.warning(f"Segmentation JSON {json_path.name} not found")
        return

    # 读取原始图片
    try:
        raw_image = Image.open(raw_img_path)
        # raw_image_np = np.array(raw_image.convert('RGB'))  # 确保为RGB格式
    except Exception as e:
        logging.error(f"Failed to load {raw_img_path.name}: {str(e)}")
        return

    # 加载分割数据
    with open(json_path, 'r') as f:
        seg_data = json.load(f)

    valid_regions = []
    sub_regions = seg_data.get("sub_regions", [])
    img_width, img_height = raw_image.size

    # 校验原始尺寸
    sam_original_size = seg_data.get("original_size", [img_height, img_width])

    if (img_height, img_width) != tuple(sam_original_size):
        logging.warning(f"Size mismatch in {raw_img_path.name} | "
                        f"SAM: {sam_original_size} vs Actual: {(img_height, img_width)}")

    for bbox in sub_regions:
        # 校验边界框格式
        if len(bbox) != 4 or any(not isinstance(v, int) for v in bbox):
            logging.debug(f"Invalid bbox format: {bbox}")
            continue

        x1, y1, x2, y2 = bbox

        # 坐标有效性检查
        if x1 >= x2 or y1 >= y2 or x2 > img_width or y2 > img_height:
            logging.debug(f"Invalid bbox coordinates: {bbox}")
            continue

        # 计算区域面积
        area = (x2 - x1) * (y2 - y1)
        if area < 50:  # 过滤过小区域
            continue
        try:
            # 直接裁剪边界框区域
            crop_img = raw_image.crop((x1, y1, x2, y2))
            preprocessed = clip_preprocess(crop_img).unsqueeze(0).to(device)
        except Exception as e:
            logging.debug(f"Cropping failed for {bbox}: {str(e)}")
            continue

        # 特征提取与相似度计算
        with torch.no_grad():
            img_features = clip_model.encode_image(preprocessed).float()
            img_features /= img_features.norm(dim=-1, keepdim=True)

            # 温度缩放参数
            temperature = 20.0
            all_scores = []
            category_list = []

            for cat, cat_features in text_features_dict.items():
                # 计算最大相似度
                similarities = (img_features @ cat_features.T) * temperature
                max_similarity = similarities.max().item()
                all_scores.append(max_similarity)
                category_list.append(cat)

                # Softmax归一化
            scores_tensor = torch.tensor(all_scores)
            probs = torch.softmax(scores_tensor, dim=0)
            max_score, best_idx = torch.max(probs, dim=0)
            best_category = category_list[best_idx.item()]
            final_score = max_score.item()

        # 打印置信度信息（保留4位小数）
        score_percent = max_score * 100

        # 动态阈值策略调整
        base_threshold = 0.25
        area_adjustment = 0.001 * np.log1p(area / 5000)  # 对大面积区域放宽阈值
        dynamic_threshold = base_threshold - area_adjustment

        # 转换为百分比形式用于比较
        dynamic_threshold_percent = dynamic_threshold * 100
        # print(
        #     f"CLIP识别 - {raw_img_path.name} | "
        #     f"区域: {bbox} | "
        #     f"类别: {best_category:<8} | "
        #     f"置信度: {score_percent:05.2f}%"
        #     f" | 动态阈值: {dynamic_threshold_percent:05.2f}%"
        #     f" | max_score: {max_score}"
        # )

        if final_score > dynamic_threshold and best_category != 'unknown':
            valid_regions.append({
                "category": best_category,
                "bbox": [x1, y1, x2, y2],
                "clip_score": round(final_score, 4),
                "area": area
            })

        # 保存筛选结果
        output_path = DATA_PROCESSED_VIS_ANN_DIR / f"{raw_img_path.stem}.json"
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "source_image": raw_img_path.name,
                    "total_regions": len(seg_data.get('regions', [])),
                    "valid_regions": len(valid_regions)
                },
                "regions": valid_regions
            }, f, indent=2)
        logging.info(f"Processed {raw_img_path.name}: {len(valid_regions)} valid regions")


if __name__ == "__main__":
    processed_count = 0
    # 处理所有原始图片
    for img_file in DATA_RAW_DIR.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            filter_with_clip(img_file)
            processed_count += 1
    print(f"Processing complete. Total processed: {processed_count} images")




