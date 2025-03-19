# models/detector.py

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from RoadConditionAI.configs import settings
import logging

class RoadSAMDetector:
    """基于SAM1的道路状况检测器"""

    def __init__(self, model_path: str = settings.SAM_MODEL_PATH):
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        # 模型初始化
        self._init_model(model_path)

        # 标签映射
        self.id2label = {
            cls_id: (en_label, zh_label)
            for cls_id, (en_label, zh_label) in settings.LABEL_MAPPING.items()
        }

        # 可视化保存路径
        self.vis_save_dir = r"D:\Myproject\RoadConditionAI\data\processed\results\vis_img"
        os.makedirs(self.vis_save_dir, exist_ok=True)
        self.image_counter = 1  # 用于生成 vis_image_x 的序号

    def _init_model(self, model_path: str):
        """加载SAM1模型并初始化自动掩码生成器"""
        try:
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            raise

    def detect(self, image_path: str) -> Tuple[List[Dict], Dict]:
        """自动检测并分割图片"""
        # 图像预处理
        img = self._preprocess_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 使用自动掩码生成器生成掩码
        masks_data = self.mask_generator.generate(img)
        masks = [mask['segmentation'] for mask in masks_data]
        scores = [mask['stability_score'] for mask in masks_data]

        # 结果解析并可视化
        results, traffic_stats = self._parse_results(img, masks, scores)

        # 可视化并保存
        self._visualize_and_save(img, results, image_path)

        return results, traffic_stats

    def _preprocess_image(self, path: str) -> np.ndarray:
        """增强型图像预处理"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"图像解码失败: {path}")

        # 高频锐化
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

        # CLAHE对比度增强
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        limg = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((limg, a, b)), cv2.COLOR_LAB2BGR)

    def _parse_results(self, img: np.ndarray, masks: List[np.ndarray], scores: List[float]) -> Tuple[List[Dict], Dict]:
        """解析掩码并提取边界框"""
        objects = []
        traffic_stats = {'regions': len(masks)}  # 简单统计分割区域数量
        h, w = img.shape[:2]

        for i, (mask, score) in enumerate(zip(masks, scores)):
            # 从掩码中提取边界框
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            x, y, w_box, h_box = cv2.boundingRect(contours[0])
            bbox = [x, y, x + w_box, y + h_box]

            # 使用改进型标签推测
            label_en, label_zh = self._infer_label(bbox, img.shape[:2])

            obj = {
                "label": label_zh,
                "bbox": bbox,  # [x_min, y_min, x_max, y_max]
                "confidence": float(score)
            }
            objects.append(obj)
            logging.info(f"区域 {i + 1}: 标签={label_zh}, 坐标={bbox}, 置信度={score:.2f}")

            # 更新交通统计
            self._update_statistics(label_en, traffic_stats)

        return objects, traffic_stats

    def _visualize_and_save(self, img: np.ndarray, objects: List[Dict], image_path: str):
        """可视化分割区域并保存"""
        vis_img = img.copy()

        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]
            confidence = obj["confidence"]
            color = (0, 255, 0)  # 绿色边界框

            # 绘制边界框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # 添加标签和置信度
            label_text = f"{label} {confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(vis_img, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 保存结果
        save_path = os.path.join(self.vis_save_dir, f"vis_image_{self.image_counter}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        logging.info(f"可视化图像已保存至: {save_path}")
        self.image_counter += 1

    def _infer_label(self, bbox: List[int], img_shape: Tuple) -> Tuple[str, str]:
        """改进型标签推测"""
        h, w = img_shape
        x_center = (bbox[0] + bbox[2]) / 2 / w
        y_center = (bbox[1] + bbox[3]) / 2 / h
        area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h)

        if area_ratio < 0.005 and (0.3 < x_center < 0.7):
            return "debris", "抛洒物"
        return "unknown", "未知"

    def _update_statistics(self, label_en: str, stats: Dict):
        """更新交通统计"""
        # 初始化统计字典
        if 'vehicles' not in stats:
            stats['vehicles'] = {'cars': 0, 'trucks': 0, 'motorcycles': 0}
        if 'events' not in stats:
            stats['events'] = {'accident': 0, 'line_damage': 0, 'debris': 0}

        if label_en in ['car', 'truck', 'motorcycle']:
            key = label_en + 's'
            stats['vehicles'][key] += 1
        elif label_en in ['accident', 'line_damage', 'debris']:
            stats['events'][label_en] += 1

    def _get_position(self, bbox: List[int], img_shape: Tuple) -> str:
        """位置描述生成"""
        h, w = img_shape
        x_center = (bbox[0] + bbox[2]) / 2 / w
        y_center = (bbox[1] + bbox[3]) / 2 / h

        vertical = "左侧" if x_center < 0.33 else "右侧" if x_center > 0.66 else "中间"
        horizontal = "远端" if y_center < 0.3 else "近端" if y_center > 0.7 else "中段"
        return f"{vertical}车道{horizontal}区域"

if __name__ == "__main__":
    detector = RoadSAMDetector("D:\\Myproject\\RoadConditionAI\\models\\vit_h.pth")
    results, stats = detector.detect("data/raw/image_1.jpg")
    print(f"检测结果: {results}")
    print(f"交通统计: {stats}")