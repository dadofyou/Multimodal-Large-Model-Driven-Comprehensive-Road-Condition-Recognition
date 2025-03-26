import clip
import torch
from PIL import Image
import numpy as np
import json
from pathlib import Path
import logging
from RoadConditionAI.configs import settings  # 导入 settings 模块

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用 settings.py 中的路径配置
DATA_RAW_DIR = settings.DATA_RAW_DIR
DATA_PROCESSED_DIR = settings.DATA_PROCESSED_DIR
DATA_PROCESSED_VIS_ANN_DIR = settings.DATA_PROCESSED_VIS_ANN_DIR
DATA_PROCESSED_VIS_ANN_IMG_DIR = settings.DATA_PROCESSED_VIS_ANN_IMG_DIR

# 创建输出目录（如果尚未创建）
DATA_PROCESSED_VIS_ANN_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_VIS_ANN_IMG_DIR.mkdir(parents=True, exist_ok=True)

# 1. 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
logger.info(f"CLIP 模型加载完成，使用设备: {device}")

# 2. 定义道路状况配置
ROAD_STATUS_CONFIG = {
    "text_prompts": {
        # 正样本描述
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
        ],
        # 负样本描述
        "negative": [
            "A picture of a modern building",
            "A photo of a utility pole standing on the roadside",
            "Roadside guardrail along the highway",
            "A photo of trees and vegetation near the road",
            "Building facades adjacent to the street",
            "Traffic signs mounted on poles",
            "A photo of a pedestrian crossing with a lamp post",
            "Roadside advertisement boards",
            "A photo of a bus stop shelter on the street",
            "A photo of parked bicycles on the curb",
            "Roadside barriers and dividers"
        ]
    }
}

# 3. 预计算文本特征
text_features_dict = {}
for category, prompts in ROAD_STATUS_CONFIG["text_prompts"].items():
    text_inputs = torch.cat([clip.tokenize(desc) for desc in prompts]).to(device)
    with torch.no_grad():
        features = clip_model.encode_text(text_inputs).float()
        features /= features.norm(dim=-1, keepdim=True)
    text_features_dict[category] = features
logger.info("文本特征预计算完成")

# 4. 定义滤波函数
def filter_with_clip(raw_img_path: Path):
    """处理单张图片的分割区域并进行 CLIP 检测"""
    json_path = DATA_PROCESSED_DIR / f"{raw_img_path.stem}_sub.json"
    if not json_path.exists():
        logger.warning(f"未找到分割 JSON 文件: {json_path.name}")
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

    for bbox in sub_regions:
        if len(bbox) != 4 or any(not isinstance(v, int) for v in bbox):
            logger.debug(f"无效边界框格式: {bbox}")
            continue
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2 or x2 > img_width or y2 > img_height:
            logger.debug(f"无效边界框坐标: {bbox}")
            continue

        area = (x2 - x1) * (y2 - y1)
        if area < 50:  # 忽略过小区域
            continue

        try:
            crop_img = raw_image.crop((x1, y1, x2, y2))
            preprocessed = clip_preprocess(crop_img).unsqueeze(0).to(device)
        except Exception as e:
            logger.debug(f"裁剪失败 {bbox}: {str(e)}")
            continue

        with torch.no_grad():
            img_features = clip_model.encode_image(preprocessed).float()
            img_features /= img_features.norm(dim=-1, keepdim=True)

            # 获取所有类别的文本特征
            category_list = list(text_features_dict.keys())
            all_text_features = torch.cat([text_features_dict[cat] for cat in category_list], dim=0)

            # 计算相似度
            logit_scale = clip_model.logit_scale.exp()
            similarities = (img_features @ all_text_features.T) * logit_scale

            # 为每个正样本类别计算最大相似度
            start_idx = 0
            all_scores = {}
            for cat in category_list:
                num_prompts = text_features_dict[cat].shape[0]
                s_cat = similarities[0, start_idx:start_idx + num_prompts].max().item()
                all_scores[cat] = s_cat
                start_idx += num_prompts

            # 只考虑正样本类别
            positive_categories = [cat for cat in category_list if cat != 'negative']
            confidences = {k: torch.sigmoid(torch.tensor(all_scores[k])).item() for k in positive_categories}

            # 筛选置信度 > 0.5 的类别
            threshold = 0.5
            detected_categories = [k for k in positive_categories if confidences[k] > threshold]
            detected_confidences = [confidences[k] for k in detected_categories]

            if detected_categories:
                valid_regions.append({
                    "bbox": [x1, y1, x2, y2],
                    "categories": detected_categories,
                    "confidence_scores": [round(conf, 4) for conf in detected_confidences],
                    "area": area
                })

    # 保存筛选结果
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

    # # 可选：保存包含有效区域的图片
    # vis_img_path = DATA_PROCESSED_VIS_ANN_IMG_DIR / raw_img_path.name
    # raw_image.save(vis_img_path)

# 5. 主函数
if __name__ == "__main__":
    processed_count = 0
    for img_file in DATA_RAW_DIR.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            filter_with_clip(img_file)
            processed_count += 1
    logger.info(f"所有图片处理完成，共处理: {processed_count} 张")