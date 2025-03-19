# RoadConditionAI/models/detector.py
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import List, Dict
from pathlib import Path
from ..configs import settings


class RoadSegmenter:
    """道路状况专用分割器"""

    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self._init_sam_model()
        self.vis_config = settings.VISUAL_CONFIG

    def _init_sam_model(self):
        """初始化优化后的SAM模型"""
        sam = sam_model_registry["vit_h"](checkpoint=str(settings.SAM_MODEL_PATH))
        sam.to(self.device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=settings.SAM_CONFIG["points_per_side"],
            pred_iou_thresh=settings.SAM_CONFIG["pred_iou_thresh"],
            stability_score_thresh=settings.SAM_CONFIG["stability_score_thresh"],
            crop_n_layers=settings.SAM_CONFIG["crop_n_layers"],
            min_mask_region_area=settings.SAM_CONFIG["min_mask_region_area"],
            box_nms_thresh=settings.SAM_CONFIG["box_nms_thresh"]
        )

    def process_image(self, image_path: str) -> dict:
        """处理单张图像并生成可视化"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无效图像: {image_path}")

            # 生成掩码
            masks = self.mask_generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # 提取边界框
            bboxes = self._get_bbox_coordinates(masks)

            # 生成可视化
            vis_img = self._visualize_regions(img.copy(), bboxes)

            return {
                "original_size": img.shape[:2],
                "sub_num": len(bboxes),
                "sub_regions": bboxes,
                "visualization": vis_img
            }

        except Exception as e:
            print(f"处理失败 {image_path}: {str(e)}")
            return {}

    def _get_bbox_coordinates(self, masks: List[Dict]) -> List[List[int]]:
        """精确提取边界框坐标"""
        bboxes = []
        for mask_data in masks:
            mask = mask_data['segmentation']
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                # 保留小区域检测结果
                if (w * h) >= settings.SAM_CONFIG["min_mask_region_area"]:
                    bboxes.append([x, y, x + w, y + h])
        return bboxes

    def _visualize_regions(self, img: np.ndarray, bboxes: List[List[int]]) -> np.ndarray:
        """可视化分割区域"""
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # 绘制边界框
            cv2.rectangle(img,
                          (x1, y1),
                          (x2, y2),
                          self.vis_config["box_color"],
                          self.vis_config["thickness"])
            # 添加区域编号
            cv2.putText(img,
                        f"{x1},{y1}",
                        (x1 + 2, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.vis_config["font_scale"],
                        self.vis_config["box_color"],
                        1)
        return img