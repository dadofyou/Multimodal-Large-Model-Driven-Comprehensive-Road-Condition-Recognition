# RoadConditionAI/models/detector.py
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import List, Dict, Optional
from pathlib import Path
from pycocotools import mask as mask_utils
import torchvision.ops
from ..configs import settings

class RoadSegmenter:
    """优化后的道路分割器（支持CPU/GPU混合加速）"""

    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self._init_sam_model()
        self._init_postprocessor()
        self.vis_config = settings.VISUAL_CONFIG

    def _init_sam_model(self):
        """初始化优化后的SAM模型"""
        sam = sam_model_registry[settings.SAM_CONFIG["model_type"]](
            checkpoint=str(settings.SAM_MODEL_PATH)
        )
        if self.device.type == "cuda":
            sam.to(self.device)
            if settings.SAM_CONFIG["use_fp16"]:
                sam = sam.half()
                torch.backends.cudnn.benchmark = True
        else:
            # 在CPU上禁用fp16
            settings.SAM_CONFIG["use_fp16"] = False

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=settings.SAM_CONFIG["points_per_side"],
            pred_iou_thresh=settings.SAM_CONFIG["pred_iou_thresh"],
            stability_score_thresh=settings.SAM_CONFIG["stability_score_thresh"],
            crop_n_layers=settings.SAM_CONFIG["crop_n_layers"],
            min_mask_region_area=settings.SAM_CONFIG["min_mask_region_area"],
            box_nms_thresh=settings.SAM_CONFIG["box_nms_thresh"],
            output_mode="coco_rle" if settings.SAM_CONFIG.get("use_rle", True) else "binary_mask"
        )

    def _init_postprocessor(self):
        """初始化后处理组件"""
        self.min_area = settings.SAM_CONFIG["min_mask_region_area"]
        self.max_aspect_ratio = 5.0  # 最大长宽比
        self.min_aspect_ratio = 0.2  # 最小长宽比

    def process_image(self, image_path: str) -> Optional[dict]:
        """优化后的图像处理流水线"""
        try:
            # 阶段1：图像加载与预处理
            img_tensor, scale = self._load_and_preprocess(image_path)
            if img_tensor is None:
                return None

            # 阶段2：生成掩码（自动设备感知）
            masks = self._generate_masks(img_tensor)

            # 阶段3：后处理与坐标转换
            bboxes = self._postprocess_masks(masks, scale)

            # 阶段4：生成可视化图像
            orig_img = cv2.imread(image_path)
            if orig_img is None:
                return None
            vis_img = self._fast_visualization(orig_img, bboxes)

            return {
                "original_size": orig_img.shape[:2],
                "sub_num": len(bboxes),
                "sub_regions": bboxes,
                "visualization": vis_img
            }
        except Exception as e:
            print(f"处理失败 {image_path}: {str(e)}")
            return None

    def _load_and_preprocess(self, image_path: str):
        """图像加载与预处理"""
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None, 1.0
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        target_size = 1024
        scale = target_size / max(h, w)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_rgb.copy()
            scale = 1.0
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).to(self.device)
        if settings.SAM_CONFIG["use_fp16"] and self.device.type == "cuda":
            img_tensor = img_tensor.half()
        else:
            img_tensor = img_tensor.float()
        return img_tensor, scale

    def _generate_masks(self, img_tensor: torch.Tensor):
        """生成掩码（支持GPU加速）"""
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        if self.device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=True):
                masks = self.mask_generator.generate(img_np)
        else:
            masks = self.mask_generator.generate(img_np)
        return masks

    def _postprocess_masks(self, masks: List[Dict], scale: float) -> List[List[int]]:
        """多阶段后处理流程"""
        valid_bboxes = []
        for mask_data in masks:
            mask = self._decode_mask(mask_data)
            if mask is None:
                continue
            if not self._filter_region(mask):
                continue
            bbox = self._convert_coordinates(mask, scale)
            valid_bboxes.append(bbox)
        return self._nms_filter(valid_bboxes)

    def _decode_mask(self, mask_data: Dict) -> Optional[np.ndarray]:
        """解码掩码数据"""
        try:
            if isinstance(mask_data['segmentation'], dict):
                return mask_utils.decode(mask_data['segmentation'])
            return mask_data['segmentation'].astype(np.uint8)
        except Exception as e:
            print(f"掩码解码失败: {str(e)}")
            return None

    def _filter_region(self, mask: np.ndarray) -> bool:
        """区域过滤条件"""
        area = mask.sum()
        if area < self.min_area:
            return False
        y, x = np.where(mask)
        if len(x) == 0 or len(y) == 0:
            return False
        width = x.max() - x.min()
        height = y.max() - y.min()
        if width == 0 or height == 0:
            return False
        aspect_ratio = height / width
        if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
            return False
        return True

    def _convert_coordinates(self, mask: np.ndarray, scale: float) -> List[int]:
        """坐标转换与缩放"""
        y, x = np.where(mask)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        return [
            int(x_min / scale),
            int(y_min / scale),
            int(x_max / scale),
            int(y_max / scale)
        ]

    def _nms_filter(self, bboxes: List[List[int]]) -> List[List[int]]:
        """NMS过滤"""
        if not bboxes:
            return []
        boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        keep_idx = torchvision.ops.nms(
            boxes_tensor,
            areas,
            settings.SAM_CONFIG["box_nms_thresh"]
        )
        return [bboxes[i] for i in keep_idx.tolist()]

    def _fast_visualization(self, img: np.ndarray, bboxes: List[List[int]]) -> np.ndarray:
        """快速生成可视化结果"""
        canvas = img.copy()
        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            cv2.rectangle(canvas, (x1, y1), (x2, y2),
                          self.vis_config["box_color"],
                          self.vis_config["thickness"])
            if self.vis_config["show_labels"]:
                label = f"{idx}"
                (tw, th), _ = cv2.getTextSize(label,
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              self.vis_config["font_scale"], 1)
                cv2.rectangle(canvas,
                              (x1, y1 - th - 2),
                              (x1 + tw, y1),
                              self.vis_config["box_color"],
                              -1)
                cv2.putText(canvas, label,
                            (x1 + 1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.vis_config["font_scale"],
                            (255, 255, 255), 1)
        return canvas
