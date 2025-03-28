import clip
import torch
from PIL import Image
import numpy as np
import json
from pathlib import Path
import logging
import shutil
import time
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
"categories": {
        "accident": {
            "positive": [
                "A clear image capturing a severe traffic accident with multiple damaged vehicles and scattered debris",
                "A vivid photo showing a collision scene on the road with crashed cars, broken glass, and emergency responders present",
                "An image depicting a chaotic traffic accident with overturned vehicles and visible damage to road infrastructure",
                "Aerial view of multiple vehicles collided with shattered windshields and deployed airbags visible",
                "Close-up of car crash showing twisted metal fragments and spilled fluids on asphalt surface",
                "Wide-angle scene with emergency vehicles flashing lights around overturned truck on highway",
                #"Nighttime collision involving motorcycles with broken plastic parts scattered across intersection",
                #"Dashcam footage capturing chain-reaction crash with flying debris and smoke emission",
                "Collision impact deforming vehicle's frame structure with bent pillars",
                "Frontal crash causing hood crumpling or airbag deployment"


            ],
            "negative": [
                #"A normal road scene with orderly traffic, clear skies, and no sign of any accident",
                "A photo of a well-maintained road with smoothly flowing traffic and no visible vehicle damage or collisions",
                "An image of a calm urban street with regular traffic and no emergency or accident-related activity",
                "Empty highway with no vehicles and perfect weather conditions",
                "Parking lot with properly parked cars and no signs of impact",
                "Car showroom displaying undamaged vehicles under studio lighting",
                "A picture showing the transported goods neatly stacked or tied with ropes with no signs of damage",
                #"Traffic junction with smooth vehicle flow and functional traffic lights",
                "Vehicle assembly line with workers installing parts on pristine car frames",
                "Delivery truck with intact body unloading packaged goods onto roadside"
            ]
        },
        "spillage": {
            "positive": [
                # 增加空间位置限定词（关键修改点）
                "Three-dimensional debris protruding above road surface texture",
                "Broken objects scattered on asphalt with shadow cast downward",
                "Foreign materials accumulated near roadside drainage gutter",
                "Spilled cargo dispersed across traffic lane without vertical structures",
                "Disposed household garbage bags torn open with plastic debris spreading",
                "Cardboard boxes burst open spilling packaged goods across multiple lanes",
                "Asphalt surface covered with nails and screws from construction vehicle spill",
                "Windblown sand dunes formed across rural road after sand truck leakage",
                #"Asymmetrical material dispersion unrelated to traffic guidance systems"
            ],
            "negative": [
                # 强化车辆部件特征（新增针对性负样本）
                "Car door panel with broken glass still attached to vehicle frame",
                "Overturned truck showing intact cargo compartment walls",
                "Vehicle roof with dented metal but no detached parts",
                "Close-up of windshield cracks spreading across glass surface",
                "Bicycle rack mounted on car roof carrying sp'orts equipment",
                "Construction materials properly secured with cargo nets on trailer 'bed",
                "Freshly painted road markings with crisp edges and high retroreflectivity",
                "Worn but intact dashed lines showing gradual fading from traffic wear",
                "Worn thermoplastic markings with partial detachment but preserved alignment",
                "Abrasion-faded symbols still following original stencil patterns"
            ]
         },
        "lane_damage": {
            "positive": [
                "Macro shot revealing cracked lane markings with paint peeling up at edges",
                "Infrared image showing deteriorated thermoplastic markings with low reflectivity",
                "Rainy scene where missing lane lines cause ambiguous road boundaries",
                "Time-lapse sequence showing progressive fading of dashed lane markings",
                "Thermal camera view exposing sub-surface delamination under damaged road lines"
            ],
            "negative": [
                "Newly applied road markings with crisp edges and high retroreflectivity",
                "Temporary construction zone with clearly visible orange thermoplastic lines",
                "Bicycle lane with fresh green paint and protective epoxy coating",
                "Road surface with shadows mimicking line damage but intact markings",
                "Wet road reflecting street lights without actual marking deterioration"
            ]
        },
        "illegal_occupation": {
            "positive": [
                "Diagonal parking completely blocking bicycle lane with hazard lights on",
                "Delivery trucks double-parked creating narrow passage for through traffic",
                "Construction equipment occupying two lanes without proper permits displayed",
                "Nighttime scene of private vehicles blocking emergency vehicle access path",
                "Farm machinery slowly moving on highway without required escort vehicles"
            ],
            "negative": [
                "Designated loading zone with commercial vehicles during permitted hours",
                "Police car properly stopped in breakdown lane with flashing roof lights",
                "Funeral procession with official authorization and proper safety controls",
                "Roadwork area with legally closed lanes and certified traffic controllers",
                "Disabled vehicle temporarily parked with triangle warning sign appropriately placed"
            ]
        },
        "pothole": {
            "positive": [
                "Cross-sectional view revealing pothole depth exceeding 10cm with exposed aggregate",
                "After-rain scene showing wheel-sized pothole filled with muddy water",
                "Thermal imaging highlighting temperature differential at pothole edges",
                "Close-up of alligator cracking pattern leading to pavement disintegration",
                "3D scan visualization showing subsurface cavity under apparent surface hole"
            ],
            "negative": [
                "Manhole cover properly installed flush with road surface",
                "Expansion joints with designed gaps filled with elastic sealant",
                "Speed bump with standardized height and warning signage",
                "Roadside drainage grate with metal surface intact",
                "Superficial asphalt discoloration without structural deformation"
            ]
        }
    },
    # FIXME:各问题的阈值可根据实验调优
    "thresholds": {
        "accident": 0.63,
        "spillage": 0.60,
        "lane_damage": 0.61,
        "illegal_occupation": 0.55,
        "pothole": 0.60
    }
}

# 3. 预计算文本特征（分别计算正描述与负描述）
text_features_dict = {}
for category, prompts in ROAD_STATUS_CONFIG["categories"].items():
    text_features_dict[category] = {}
    # 预计算正描述特征
    positive_prompts = prompts["positive"]
    positive_inputs = torch.cat([clip.tokenize(desc) for desc in positive_prompts]).to(device)
    with torch.no_grad():
        pos_features = clip_model.encode_text(positive_inputs).float()
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
    text_features_dict[category]["positive"] = pos_features

    # 预计算负描述特征
    negative_prompts = prompts["negative"]
    negative_inputs = torch.cat([clip.tokenize(desc) for desc in negative_prompts]).to(device)
    with torch.no_grad():
        neg_features = clip_model.encode_text(negative_inputs).float()
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
    text_features_dict[category]["negative"] = neg_features

logger.info("文本特征预计算完成")


def compute_iou(bbox1, bbox2):
    """计算两个边界框的交并比 (IoU)"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def merge_overlapping_regions(regions, iou_threshold=0.3):
    """合并高度重叠的区域"""
    merged = []
    used = [False] * len(regions)

    for i in range(len(regions)):
        if used[i]:
            continue
        # 当前聚类的初始区域
        cluster = [regions[i]]
        used[i] = True

        for j in range(i + 1, len(regions)):
            if used[j]:
                continue
            iou = compute_iou(regions[i]["bbox"], regions[j]["bbox"])
            if iou >= iou_threshold:
                cluster.append(regions[j])
                used[j] = True

        # 合并聚类中的区域：合并边界框为并集
        xs = [r["bbox"][0] for r in cluster]
        ys = [r["bbox"][1] for r in cluster]
        xe = [r["bbox"][2] for r in cluster]
        ye = [r["bbox"][3] for r in cluster]
        merged_bbox = [min(xs), min(ys), max(xe), max(ye)]

        # 合并类别：取所有区域中每个类别的最大置信度
        merged_categories = {}
        for r in cluster:
            for cat, conf in zip(r["categories"], r["confidence_scores"]):
                if cat in merged_categories:
                    merged_categories[cat] = max(merged_categories[cat], conf)
                else:
                    merged_categories[cat] = conf

        merged_region = {
            "bbox": merged_bbox,
            "categories": list(merged_categories.keys()),
            "confidence_scores": [round(merged_categories[cat], 4) for cat in merged_categories],
            "area": (merged_bbox[2] - merged_bbox[0]) * (merged_bbox[3] - merged_bbox[1])
        }
        merged.append(merged_region)
    return merged


def filter_with_clip(raw_img_path: Path):
    """处理单张图片的分割区域并进行 CLIP 粗检测（引入缩放因子以提高置信度动态范围）"""
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

    # 调整面积过滤阈值为 1000（可以根据实际情况进一步调优）
    min_area_threshold = 500
    # 定义缩放因子，用于放大相似度差值
    scaling_factor = 10.0

    for bbox in sub_regions:
        if len(bbox) != 4 or any(not isinstance(v, int) for v in bbox):
            logger.debug(f"无效边界框格式: {bbox}")
            continue
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2 or x2 > img_width or y2 > img_height:
            logger.debug(f"无效边界框坐标: {bbox}")
            continue

        area = (x2 - x1) * (y2 - y1)
        if area < min_area_threshold:
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
            region_detected_categories = []
            region_confidences = []

            # 针对每个问题进行判断
            for category in ROAD_STATUS_CONFIG["categories"].keys():
                # 正样本相似度，使用均值
                pos_features = text_features_dict[category]["positive"]
                pos_sim = (img_features @ pos_features.T)
                pos_sim_mean = pos_sim.mean().item()
                # 负样本相似度，使用均值
                neg_features = text_features_dict[category]["negative"]
                neg_sim = (img_features @ neg_features.T)
                neg_sim_mean = neg_sim.mean().item()
                # 计算差值并引入缩放因子
                diff = scaling_factor * (pos_sim_mean - neg_sim_mean)
                # 经过 sigmoid 映射到 0-1 范围
                confidence = torch.sigmoid(torch.tensor(diff)).item()

                # 根据各类别设置的阈值判断是否存在该问题
                threshold = ROAD_STATUS_CONFIG["thresholds"][category]
                if confidence > threshold:
                    region_detected_categories.append(category)
                    region_confidences.append(round(confidence, 4))

            if region_detected_categories:
                valid_regions.append({
                    "bbox": [x1, y1, x2, y2],
                    "categories": region_detected_categories,
                    "confidence_scores": region_confidences,
                    "area": area
                })

    # 对检测出的区域进行合并，减少重叠区域（IOU 阈值设为 0.3）
    merged_regions = merge_overlapping_regions(valid_regions, iou_threshold=0.3)

    # 保存筛选结果
    output_path = DATA_PROCESSED_VIS_ANN_DIR / f"{raw_img_path.stem}.json"
    output_data = {
        "metadata": {
            "source_image": raw_img_path.name,
            "total_regions": len(sub_regions),
            "valid_regions": len(merged_regions)
        },
        "regions": merged_regions
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"处理完成 {raw_img_path.name}: {len(merged_regions)} 个有效区域")
    # 可选：保存包含有效区域的图片（如果需要可取消注释）
    # vis_img_path = DATA_PROCESSED_VIS_ANN_IMG_DIR / f"{raw_img_path.name}"
    # raw_image.save(vis_img_path)


# 5. 主函数
# ... 此处省略前面已有的代码（包括 filter_with_clip 函数等） ...

if __name__ == "__main__":
    from RoadConditionAI.utils.clip_visualize import visualize_detection

    processed_count = 0
    # 这里取前 5 张图片进行测试
    for img_file in list(DATA_RAW_DIR.glob('*'))[20:30]:
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # 先进行检测并生成 JSON 文件
            filter_with_clip(img_file)
            # 构造 JSON 文件路径（与 filter_with_clip 中保存的保持一致）
            json_path = DATA_PROCESSED_VIS_ANN_DIR / f"{img_file.stem}.json"
            # 调用可视化函数生成检测结果图片
            vis_path = visualize_detection(img_file, json_path, DATA_PROCESSED_VIS_ANN_IMG_DIR)
            if vis_path:
                print(f"生成结果：{vis_path}")
            else:
                print("生成失败")
            processed_count += 1
    logger.info(f"所有图片处理完成，共处理: {processed_count} 张")
