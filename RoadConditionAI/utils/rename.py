# RoadConditionAI/utils/rename.py
import logging
from pathlib import Path
import cv2
import numpy as np
from ..configs import settings


def rename_images():
    """统一重命名并转换图片格式"""
    try:
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )

        # 获取所有图片文件
        image_files = sorted(
            settings.DATA_RAW_UNFILTERED.glob("*.*"),
            key=lambda x: x.name.lower()
        )

        # 筛选有效图片格式
        valid_files = [
            f for f in image_files
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]

        # 转换和重命名处理
        for idx, file_path in enumerate(valid_files):
            try:
                # PNG转JPG处理
                if file_path.suffix.lower() == ".png":
                    img = cv2.imread(str(file_path))
                    if img is None:
                        raise ValueError("无效的PNG文件")

                    # 删除原PNG文件
                    file_path.unlink()

                    # 生成新的JPG路径
                    new_path = file_path.with_suffix(".jpg")
                    cv2.imwrite(
                        str(new_path),
                        img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                    )
                    logging.info(f"转换PNG到JPG: {file_path.name} -> {new_path.name}")
                else:
                    new_path = file_path

                # 统一重命名
                target_path = settings.DATA_RAW_UNFILTERED / f"image_{idx}.jpg"

                # 处理已存在的目标文件
                if target_path.exists():
                    target_path.unlink()

                new_path.rename(target_path)
                logging.info(f"重命名成功: {new_path.name} -> image_{idx}.jpg")

            except Exception as e:
                logging.error(f"处理失败 {file_path.name}: {str(e)}")
                continue

        logging.info(f"共处理 {len(valid_files)} 张图片")

    except Exception as e:
        logging.error(f"重命名流程异常: {str(e)}")
        raise