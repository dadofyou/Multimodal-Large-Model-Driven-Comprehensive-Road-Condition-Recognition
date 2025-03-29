# RoadConditionAI/configs/settings.py
from pathlib import Path
import torch

# 项目路径配置
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_RAW_UNFILTERED = BASE_DIR / "data" / "raw" / "img_unfiltered"

# 数据目录配置
DATA_RAW_DIR = BASE_DIR / "data" / "raw" / "img_unfiltered"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed" / "results" / "img_division"
DATA_PROCESSED_VIS_DIR = BASE_DIR / "data" / "processed" / "results" / "img_division_vis"
DATA_PROCESSED_VIS_ANN_DIR = BASE_DIR / "data" / "processed" / "results" / "annotations"
DATA_PROCESSED_VIS_ANN_IMG_DIR = BASE_DIR / "data" / "processed" / "results" / "ann_img_img"

# SAM模型路径
SAM_MODEL_PATH = BASE_DIR / "models" / "vit_h.pth"

# 设备配置（支持GPU和CPU）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAM配置（多尺度切割，覆盖大区域与小区域）
SAM_CONFIG = {
    "model_type": "vit_h",
    "use_rle": True,             # 启用RLE压缩
    "use_fp16": True,            # 启用混合精度（CPU上会自动禁用）
    "points_per_side": 24,       # 增大采样密度，便于均匀覆盖大区域和小区域
    "pred_iou_thresh": 0.80,     # 稍微降低IoU阈值，以保留大面积区域
    "stability_score_thresh": 0.85,
    "crop_n_layers": 1,          # 增加分层检测，保证多尺度分割
    "min_mask_region_area": 500, # 调整最小区域面积（适当降低以便捕获更多细节）
    "box_nms_thresh": 0.7,
    "crop_n_points_downscale_factor": 2
}

# 性能配置
PERF_CONFIG = {
    "max_workers": 4,            # 根据CPU核心数调整
    "gpu_chunk_size": 1024       # 显存优化参数
}

# 可视化配置
VISUAL_CONFIG = {
    "box_color": (0, 255, 0),    # BGR绿色边框
    "thickness": 1,              # 边框线条粗细
    "font_scale": 0.4,           # 文字大小
    "show_labels": True          # 控制是否显示标签
}

# 创建目录（包含可视化目录）
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_VIS_DIR.mkdir(parents=True, exist_ok=True)