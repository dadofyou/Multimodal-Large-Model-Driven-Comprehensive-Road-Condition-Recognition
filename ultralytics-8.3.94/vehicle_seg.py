from ultralytics import YOLO
import cv2
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np


class VehicleCropper:
    def __init__(self):
        # 初始化基准路径
        self.base_dir = Path("../RoadConditionAI/data/processed/results")

        # 输入输出路径配置
        self.raw_img_dir = Path("../RoadConditionAI/data/raw/img_unfiltered")
        self.output_vis_dir = self.base_dir / "yolo_img_division_vis"
        self.output_json_dir = self.base_dir / "yolo_img_division"

        # 创建输出目录
        self.output_vis_dir.mkdir(parents=True, exist_ok=True)
        self.output_json_dir.mkdir(parents=True, exist_ok=True)

        # 初始化模型
        self.det_model = YOLO('yolo11l.pt')
        self.class_names = self.det_model.names

    # 动态置信度计算
    def calculate_dynamic_conf(self, height, width,
                               base_size=640,  # 基准分辨率（训练尺寸）
                               base_conf=0.43,  # 基准置信度
                               min_conf=0.38,  # 最小置信度
                               max_conf=0.6):  # 最大置信度
        """
        基于图像短边长度的动态置信度计算
        计算逻辑：
        - 当图像短边 >= base_size时：使用base_conf
        - 当图像短边 < base_size时：线性下降到min_conf
        """
        short_side = min(height, width)
        scale = min(1.0, short_side / base_size)

        # 线性插值公式
        dynamic_conf = base_conf * scale + min_conf * (1 - scale)

        # 限制在[min_conf, max_conf]范围内
        return np.clip(dynamic_conf, min_conf, max_conf)

    def process_image(self, img_name="image_0.jpg"):
        """处理单张图片"""
        # 输入输出路径
        img_path = self.raw_img_dir / img_name
        base_name = img_path.stem

        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        H, W = img.shape[:2]

        # 初始化结果结构
        result = {
            "original_size": [
                H,
                W
            ],
            "sub_num": 0,
            "vehicles": []
        }

        dynamic_conf = self.calculate_dynamic_conf(H, W)

        # 执行检测
        det_results = self.det_model.predict(img, classes=[2, 5, 7], conf=dynamic_conf)

        # 处理每个检测结果
        for i, box in enumerate(det_results[0].boxes):
            # 获取基础信息
            xyxy = box.xyxy[0].tolist()
            conf = box.conf.item()
            cls_id = int(box.cls.item())

            # 扩展裁剪区域
            x1, y1, x2, y2 = self._expand_roi(xyxy, W, H, expand_ratio=0.1)

            # 保存裁剪图像
            crop_img = img[y1:y2, x1:x2]
            vis_dir = self.output_vis_dir / f"{base_name}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_path = vis_dir / f"vehicle_{i}.jpg"
            cv2.imwrite(str(vis_path), crop_img)

            # 添加到结果
            result["vehicles"].append({
                "vehicle_id": f"{base_name}_vehicle_{i}",
                "class_id": cls_id,
                "class_name": self.class_names[cls_id],
                "confidence": round(conf, 4),
                "original_bbox": [round(x, 2) for x in xyxy],
                "expanded_bbox": [x1, y1, x2, y2],
                "crop_image_path": str(vis_path.relative_to(self.output_vis_dir))
            })

        # 保存JSON结果
        json_path = self.output_json_dir / f"{base_name}_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return json_path

    def _expand_roi(self, xyxy, img_w, img_h, expand_ratio=0.2):
        """扩展检测框区域"""
        x1, y1, x2, y2 = map(int, xyxy)

        # 计算扩展量
        w = x2 - x1
        h = y2 - y1
        expand_w = int(w * expand_ratio)
        expand_h = int(h * expand_ratio)

        # 应用扩展
        new_x1 = max(0, x1 - expand_w)
        new_y1 = max(0, y1 - expand_h)
        new_x2 = min(img_w, x2 + expand_w)
        new_y2 = min(img_h, y2 + expand_h)

        return new_x1, new_y1, new_x2, new_y2


if __name__ == "__main__":
    processor = VehicleCropper()

    for img_file in processor.raw_img_dir.glob("*.jpg"):
        processor.process_image(img_file.name)
