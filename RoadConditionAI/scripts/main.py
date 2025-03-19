# scripts/main.py

import os
import sys
import json
import logging
import random
import cv2
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from PIL import Image, ExifTags
from RoadConditionAI.configs import settings
from pathlib import Path
from RoadConditionAI.models.detector import RoadSAMDetector  # 导入 SAM 检测器
from RoadConditionAI.scripts.sub_img_rec import classify_with_clip

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)


class Visualizer:
    """增强型可视化工具（网页5的样式改进）"""
    COLOR_PALETTE = {
        "交通事故": (0, 0, 255),      # 红色
        "标志线损坏": (255, 0, 0),     # 蓝色
        "抛洒物": (0, 165, 255),       # 橙色
        "default": (0, 255, 0)         # 绿色
    }

    @classmethod
    def draw_boxes(cls, image: np.ndarray, boxes: list) -> np.ndarray:
        """绘制检测框和标签"""
        for box in boxes:
            x1, y1, x2, y2 = map(int, box["bbox"])
            label = box["label"]
            color = cls.COLOR_PALETTE.get(label, cls.COLOR_PALETTE["default"])

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 添加标签文本
            label_text = f"{label} {box['confidence']:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(image, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return image

def validate_settings():
    """验证必要配置参数"""
    required_settings = [
        'SAM_MODEL_PATH', 'DATA_RAW_DIR', 'DATA_PROCESSED_DIR',
        'ROAD_STATUS_MAPPING', 'DEVICE', 'VEHICLE_CLASSES'
    ]
    missing = [s for s in required_settings if not hasattr(settings, s)]
    if missing:
        raise RuntimeError(f"缺失必要配置参数: {missing}")

def initialize_detector() -> RoadSAMDetector:
    """初始化 SAM 检测器"""
    try:
        detector = RoadSAMDetector(settings.SAM_MODEL_PATH)
        logging.info(f"使用设备: {detector.device}")
        return detector
    except Exception as e:
        logging.error(f"检测器初始化失败: {str(e)}")
        raise

def update_road_status(objects: List[Dict]) -> Dict:
    """更新道路状态标志"""
    status = {
        "has_litter": False,
        "has_line_damage": False,
        "has_pothole": False,
        "has_accident": False,
        "has_illegal_occupancy": False
    }

    for obj in objects:
        label = obj["label"]
        confidence = obj.get("confidence", 0)

        # 抛洒物检测（高置信度要求）
        if label == "抛洒物" and confidence > 0.7:
            status["has_litter"] = True

        # 标志线损坏检测
        if label == "标志线损坏":
            status["has_line_damage"] = True
            # 计算损坏面积占比
            bbox = obj.get("metadata", {}).get("normalized_bbox", [0, 0, 0, 0])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width * height > 0.1:  # 面积超过10%
                status["has_line_damage"] = "severe"

        # 坑槽检测
        if label == "坑槽":
            status["has_pothole"] = True

        # 交通事故检测
        if label == "交通事故":
            status["has_accident"] = True

        # 违规占道检测（右侧区域）
        if label == "违规占道" and obj.get("position", "").startswith("右侧"):
            status["has_illegal_occupancy"] = True

    return status

def get_exif_data(img_path: str) -> Dict:
    """提取EXIF元数据"""
    try:
        img = Image.open(img_path)
        exif = img._getexif() or {}
        return {
            ExifTags.TAGS[k]: v
            for k, v in exif.items()
            if k in ExifTags.TAGS
        }
    except Exception as e:
        logging.warning(f"无法提取EXIF数据: {str(e)}")
        return {}

def process_image_metadata(img_path: str) -> Dict:
    """图像元数据处理"""
    try:
        img = cv2.imread(img_path)
        return {
            "resolution": f"{img.shape[1]}x{img.shape[0]}" if img is not None else "Unknown",
            "size": f"{os.path.getsize(img_path)/1024:.2f} KB",
            "modified_time": datetime.fromtimestamp(os.path.getmtime(img_path)).isoformat(),
            "exif": get_exif_data(img_path)
        }
    except Exception as e:
        logging.error(f"元数据处理失败: {str(e)}")
        return {}

def generate_road_analysis(objects: List[Dict]) -> str:
    """生成道路状况分析报告"""
    accident_count = sum(1 for o in objects if o["label"] == "交通事故")
    damage_count = sum(1 for o in objects if "损坏" in o["label"])

    analysis = []
    if accident_count > 0:
        types = [o["type"] for o in objects if o["label"] == "交通事故"]
        main_type = max(set(types), key=types.count) if types else "未知类型"
        analysis.append(f"检测到{accident_count}起交通事故（主要类型：{main_type}）")

    if damage_count > 0:
        analysis.append(f"发现{damage_count}处道路设施损坏")

    return "；".join(analysis) if analysis else "道路状况正常"

def generate_qa_pairs(objects: List[Dict], road_status: Dict) -> List[Dict]:
    """生成问答对"""
    qa_pairs = []

    # 状态类问答
    for key, config in settings.ROAD_STATUS_MAPPING.items():
        status = road_status.get(key, False)
        if status:
            answer = f"检测到{config['positive_answer']}"
            if key == "has_line_damage" and status == "severe":
                answer += "，损坏面积超过10%"
            qa_pairs.append({"question": config["question"], "answer": answer})
        elif random.random() < settings.NEGATIVE_QA_PROB:
            qa_pairs.append({
                "question": config["question"],
                "answer": f"当前未发现{config['negative_answer']}"
            })

    # 统计类问答
    if objects:
        avg_conf = sum(o["confidence"] for o in objects) / len(objects)
        qa_pairs.append({
            "question": "检测置信度如何？",
            "answer": f"平均检测置信度：{avg_conf:.2f}（范围：{min(o['confidence'] for o in objects):.2f}-{max(o['confidence'] for o in objects):.2f}）"
        })

    # 系统信息类问答（改为 SAM 模型版本信息）
    qa_pairs.extend([
        {
            "question": "使用的检测模型版本",
            "answer": f"SAM 模型权重文件: {os.path.basename(settings.SAM_MODEL_PATH)}"
        },
        {
            "question": "道路状况综合分析",
            "answer": generate_road_analysis(objects)
        }
    ])

    return qa_pairs

def process_image(filename: str, detector: RoadSAMDetector) -> Optional[Dict]:
    """核心处理流程"""
    try:
        img_path = Path(settings.DATA_RAW_DIR) / filename
        if not img_path.exists():
            logging.warning(f"文件不存在: {img_path}")
            return None

        # 执行检测
        objects, traffic_stats = detector.detect(str(img_path))

        # 生成可视化结果
        img = cv2.imread(str(img_path))
        if img is not None and objects:
            vis_img = Visualizer.draw_boxes(img.copy(), objects)
            output_dir = Path(settings.DATA_PROCESSED_DIR) / (
                "accidents" if any(o["label"] == "交通事故" for o in objects) else "normal")
            output_dir.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(output_dir / f"vis_{filename}"), vis_img)

            # # 子图保存逻辑
            # segments_dir = output_dir / "segments"
            # segments_dir.mkdir(exist_ok=True)
            #
            # base_name = Path(filename).stem
            # for idx, obj in enumerate(objects):
            #     # 边界框安全校验
            #     x1, y1, x2, y2 = map(int, obj["bbox"])
            #     h, w = img.shape[:2]
            #
            #     # 裁剪区域有效性验证
            #     x1 = max(0, min(x1, w - 1))
            #     y1 = max(0, min(y1, h - 1))
            #     x2 = max(x1 + 1, min(x2, w))
            #     y2 = max(y1 + 1, min(y2, h))
            #
            #     # 执行裁剪
            #     cropped = img[y1:y2, x1:x2]
            #     if cropped.size == 0:
            #         logging.warning(f"无效区域跳过: {obj['bbox']} in {filename}")
            #         continue
            #
            #     # 调用CLIP方法进行检测。
            #     clip_result = classify_with_clip(
            #         cropped_image=cropped,
            #         subtype_mapping=settings.SUBTYPE_MAPPING
            #     )
            #
            #     # 更新对象的分类结果
            #     if clip_result:
            #         obj.update(clip_result)  # 添加 parent_label 和 subtype
            #     else:
            #         # 设置默认标签
            #         obj["parent_label"] = "未知"
            #         obj["subtype"] = "未知"
            #
            #     # 生成带中文标签的文件名
            #     safe_label = obj["label"]  # 去除非法字符
            #     seg_filename = f"{base_name}_seg{idx}_{safe_label}.jpg"
            #
            #     # 保存子图（添加JPEG质量参数）
            #     cv2.imwrite(
            #         str(segments_dir / seg_filename),
            #         cropped,
            #         [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            #     )

        # 获取元数据和道路状态
        metadata = process_image_metadata(str(img_path))
        road_status = update_road_status(objects)
        qa_pairs = generate_qa_pairs(objects, road_status)

        return {
            "image_id": Path(filename).stem,
            "detections": objects,
            "traffic_stats": traffic_stats,
            "image_size": f"{img.shape[1]}x{img.shape[0]}" if img is not None else "Unknown",
            "metadata": metadata,
            "road_status": road_status,
            "qa_pairs": qa_pairs
        }
    except Exception as e:
        logging.error(f"处理失败 {filename}: {str(e)}")
        return None


def main():
    """主流程"""
    try:
        validate_settings()
        logging.info("正在初始化 SAM 检测器...")
        detector = initialize_detector()

        processed_count = 0
        raw_dir = Path(settings.DATA_RAW_DIR)
        for img_file in raw_dir.glob(f"image_1.png"):
            logging.info(f"正在处理: {img_file.name}")
            result = process_image(img_file.name, detector)

            if result:
                output_path = Path(settings.DATA_PROCESSED_DIR) / "results" / f"{result['image_id']}.json"
                output_path.parent.mkdir(exist_ok=True, parents=True)
                with open(output_path, "w", encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                processed_count += 1

        summary = {
            "processed_images": processed_count,
            "processing_time": datetime.now().isoformat(),
            "model": "SAM",
            "device": detector.device
        }
        report_path = Path(settings.DATA_PROCESSED_DIR) / "summary.json"
        with open(report_path, "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logging.info(f"处理完成！共处理 {processed_count} 张图像，结果保存在 {settings.DATA_PROCESSED_DIR}")
    except Exception as e:
        logging.error(f"主流程异常: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
