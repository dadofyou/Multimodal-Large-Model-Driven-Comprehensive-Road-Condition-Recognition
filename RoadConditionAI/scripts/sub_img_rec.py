import logging
from pathlib import Path
import torch
import json
import clip
import numpy as np
from PIL import Image
from RoadConditionAI.configs.settings import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_VIS_ANN_DIR,
    DATA_PROCESSED_VIS_ANN_IMG_DIR
)

# 确保输出目录存在
DATA_PROCESSED_VIS_ANN_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_VIS_ANN_IMG_DIR.mkdir(parents=True, exist_ok=True)

# 设置日志输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
CLIP 环境配置：
安装依赖：pip install ftfy regex
安装 CLIP：pip install git+https://github.com/openai/CLIP.git
"""

# 初始化 CLIP 模型（兼容CPU与GPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

"""
定义道路检测语义库：
目标类别包括：抛洒物、道路标志线损坏、道路坑洞、交通事故、违规占道
每个类别给出多种描述，尽量覆盖不同角度和特征
"""
ROAD_STATUS_CONFIG = {
    "text_prompts": {
        "spillage": [
            "A photo of spilled substances on the road",
            "Road litter with scattered debris",
            "Loose materials spilled on asphalt",
            "An image showing road spill with mixed debris"
        ],
        "lane_damage": [
            "A photo of damaged or faded road markings",
            "Road with broken lane lines",
            "An image showing deteriorated road sign lines",
            "Damaged painted lines on a road surface"
        ],
        "pothole": [
            "A photo of a pothole on a road",
            "Road surface with a deep pothole",
            "An image showing a large pothole on asphalt",
            "A cracked road featuring a pothole"
        ],
        "accident": [
            "A photo of a traffic accident scene",
            "Road accident with damaged vehicles",
            "An image showing a collision on the road",
            "Traffic accident with vehicle debris on the road"
        ],
        "illegal_occupation": [
            "A photo of illegally parked vehicles obstructing the road",
            "Road occupation by vehicles not following rules",
            "An image showing vehicles parked illegally on the road",
            "Vehicles occupying road space against regulations"
        ]
    },
    # 根据实际情况可以调整各类别权重（和数量匹配）
    "category_weights": {
        "spillage": 0.25,
        "lane_damage": 0.25,
        "pothole": 0.25,
        "accident": 0.25,
        "illegal_occupation": 0.25,
    }
}

# 预计算文本特征（带类别权重）
text_features_dict = {}
for category, prompts in ROAD_STATUS_CONFIG["text_prompts"].items():
    # 每条描述都拼接上前缀 "a photo of ..." 有助于 CLIP 更好理解图像内容
    text_inputs = torch.cat([clip.tokenize(desc) for desc in prompts]).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(text_inputs).float()
        features /= features.norm(dim=-1, keepdim=True)  # 归一化
        weighted_features = features * ROAD_STATUS_CONFIG["category_weights"][category]
    text_features_dict[category] = weighted_features


def filter_with_clip(raw_img_path: Path):
    """
    处理单张图片：
      1. 读取对应的 SAM 分割 JSON 文件
      2. 加载原始图片
      3. 遍历所有分割区域，裁剪后使用 CLIP 进行图文匹配
      4. 采用动态阈值过滤无效区域，保存检测结果
    """
    json_path = DATA_PROCESSED_DIR / f"{raw_img_path.stem}_sub.json"
    if not json_path.exists():
        logger.warning(f"未找到分割JSON文件: {json_path.name}")
        return

    try:
        raw_image = Image.open(raw_img_path).convert("RGB")
    except Exception as e:
        logger.error(f"加载图片失败 {raw_img_path.name}: {str(e)}")
        return

    # 加载分割数据
    with open(json_path, 'r') as f:
        seg_data = json.load(f)

    valid_regions = []
    sub_regions = seg_data.get("sub_regions", [])
    img_width, img_height = raw_image.size
    # SAM 原始尺寸信息（一般为 [height, width]）
    sam_original_size = seg_data.get("original_size", [img_height, img_width])
    if (img_height, img_width) != tuple(sam_original_size):
        logger.warning(f"尺寸不匹配 {raw_img_path.name}: SAM {sam_original_size} vs 实际 {(img_height, img_width)}")

    for bbox in sub_regions:
        if len(bbox) != 4 or any(not isinstance(v, int) for v in bbox):
            logger.debug(f"无效边界框格式: {bbox}")
            continue
        x1, y1, x2, y2 = bbox
        # 边界坐标有效性检查
        if x1 >= x2 or y1 >= y2 or x2 > img_width or y2 > img_height:
            logger.debug(f"无效边界框坐标: {bbox}")
            continue

        area = (x2 - x1) * (y2 - y1)
        if area < 50:  # 忽略过小区域
            continue

        try:
            # 裁剪区域图像
            crop_img = raw_image.crop((x1, y1, x2, y2))
            preprocessed = clip_preprocess(crop_img).unsqueeze(0).to(device)
        except Exception as e:
            logger.debug(f"裁剪失败 {bbox}: {str(e)}")
            continue

        with torch.no_grad():
            img_features = clip_model.encode_image(preprocessed).float()
            img_features /= img_features.norm(dim=-1, keepdim=True)

            # 温度缩放参数（可根据实验调整）
            temperature = 20.0
            all_scores = []
            category_list = []

            for cat, cat_features in text_features_dict.items():
                # 计算 CLIP 相似度，取最大值作为该类别的评分
                similarities = (img_features @ cat_features.T) * temperature
                max_similarity = similarities.max().item()
                all_scores.append(max_similarity)
                category_list.append(cat)

            scores_tensor = torch.tensor(all_scores)
            probs = torch.softmax(scores_tensor, dim=0)
            max_prob, best_idx = torch.max(probs, dim=0)
            best_category = category_list[best_idx.item()]
            final_score = max_prob.item()

        # 动态阈值策略：基础阈值可根据区域面积适当放宽
        base_threshold = 0.25
        area_adjustment = 0.001 * np.log1p(area / 5000)  # 对大面积区域稍微放宽阈值
        dynamic_threshold = base_threshold - area_adjustment

        if final_score > dynamic_threshold and best_category != 'unknown':
            valid_regions.append({
                "category": best_category,
                "bbox": [x1, y1, x2, y2],
                "clip_score": round(final_score, 4),
                "area": area
            })

    # 保存筛选结果（整个图片处理完成后保存一次）
    output_path = DATA_PROCESSED_VIS_ANN_DIR / f"{raw_img_path.stem}.json"
    output_data = {
        "metadata": {
            "source_image": raw_img_path.name,
            "total_regions": len(sub_regions),
            "valid_regions": len(valid_regions)
        },
        "regions": valid_regions
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"处理完成 {raw_img_path.name}: {len(valid_regions)} 个有效区域")

    # 可选：将包含有效区域的图片保存到指定目录（标注可视化）
    # 此处仅保存原图，可以扩展为在图像上标注 bbox 后保存
    vis_img_path = DATA_PROCESSED_VIS_ANN_IMG_DIR / raw_img_path.name
    raw_image.save(vis_img_path)


if __name__ == "__main__":
    processed_count = 0
    for img_file in DATA_RAW_DIR.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            filter_with_clip(img_file)
            processed_count += 1
    logger.info(f"所有图片处理完成，共处理: {processed_count} 张")





