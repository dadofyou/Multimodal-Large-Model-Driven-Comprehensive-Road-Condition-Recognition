# configs/settings.py

import os
from typing import Dict, List, Tuple
import torch

# 项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 数据目录设置
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CORPUS_FILENAME = "corpus.json"

# 模型设置（使用 SAM1 模型权重）
SAM_MODEL_PATH = os.path.join(BASE_DIR, "models", "vit_h.pth")

# 检测参数
DETECTION_CONFIDENCE = 0.6
IMG_TARGET_SIZE = (1024, 1024)

LABEL_MAPPING: Dict[int, Tuple[str, str]] = {
    0: ("car", "机动车"),
    1: ("truck", "货车"),
    2: ("accident", "交通事故"),
    3: ("dispersion", "抛洒物"),
    4: ("line_damage", "标志线损坏"),
    5: ("pothole", "坑槽"),
    6: ("motorcycle", "摩托车"),
    7: ("illegal_occupancy", "违规占道")
}

# 类型细分映射
SUBTYPE_MAPPING: Dict[str, List[str]] = {
    "交通事故": ["追尾", "侧碰", "刮蹭", "翻车", "多车连撞"],
    "标志线损坏": ["车道线模糊", "标志缺失", "反光失效", "标线磨损", "标线覆盖"],
    "抛洒物": ["石块", "货物散落", "垃圾堆积", "油污泄漏", "动物尸体"],
    "违规占道": ["应急车道占用", "施工占道", "违法停车", "违规变道", "路障设置"]
}

# 负样本 QA 对生成概率
NEGATIVE_QA_PROB = 0.3

# 道路状态与问答模板
ROAD_STATUS_MAPPING = {
    "has_litter": {
        "question": "图中公路上有没有抛洒物？",
        "positive_answer": "存在抛洒物。",
        "negative_answer": "不存在抛洒物。"
    },
    "has_line_damage": {
        "question": "图中公路上有没有标志线损坏？",
        "positive_answer": "存在标志线损坏。",
        "negative_answer": "不存在标志线损坏。"
    },
    "has_pothole": {
        "question": "图中公路上是否存在坑槽？",
        "positive_answer": "存在坑槽。",
        "negative_answer": "不存在坑槽。"
    },
    "has_accident": {
        "question": "图中公路上有没有交通事故发生？",
        "positive_answer": "存在交通事故。",
        "negative_answer": "不存在交通事故。"
    },
    "has_illegal_occupancy": {
        "question": "图中公路上有没有违规占道？",
        "positive_answer": "存在违规占道。",
        "negative_answer": "不存在违规占道。"
    }
}

# 天气信息
DEFAULT_WEATHER = "晴天"

# 车辆类别
VEHICLE_CLASSES = ["机动车", "货车", "摩托车"]  # 使用中文标签，与 LABEL_MAPPING 一致

# 设备选择（支持动态切换）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 动态检测
