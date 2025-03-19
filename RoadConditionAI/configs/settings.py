# RoadConditionAI/configs/settings.py
from pathlib import Path
import torch

# 项目路径配置
BASE_DIR = Path(__file__).parent.parent.resolve()
# 添加原始图片路径配置
DATA_RAW_UNFILTERED = Path(__file__).parent.parent / "data" / "raw" / "img_unfiltered"

# 数据目录配置
DATA_RAW_DIR = BASE_DIR / "data" / "raw" / "img_unfiltered"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed" / "results" / "img_division"
DATA_PROCESSED_VIS_DIR = BASE_DIR / "data" / "processed" / "results" / "img_division_vis"  # 新增可视化目录
# SAM模型路径
SAM_MODEL_PATH = BASE_DIR / "models" / "vit_h.pth"

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAM优化参数（基于道路检测特点调整）
SAM_CONFIG = {
    "points_per_side": 48,        # 增加采样密度以检测小物体
    "pred_iou_thresh": 0.8,       # 降低IoU阈值保留潜在目标
    "stability_score_thresh": 0.85, # 平衡稳定性和灵敏度
    "crop_n_layers": 2,           # 增加分层检测
    "min_mask_region_area": 50,   # 减小最小区域面积
    "crop_n_points_downscale_factor": 1,
    "box_nms_thresh": 0.7
}

# 可视化参数
VISUAL_CONFIG = {
    "box_color": (0, 255, 0),     # BGR绿色边框
    "thickness": 1,               # 框线粗细
    "font_scale": 0.4            # 文字大小
}

# 创建目录（添加可视化目录）
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_VIS_DIR.mkdir(parents=True, exist_ok=True)