# RoadConditionAI/utils/rename.py
import logging
import re
from pathlib import Path
import cv2
import numpy as np
from ..configs import settings


def rename_images():
    """完全重新编号所有图片文件（安全版本）"""
    try:
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )

        # 第一步：转换所有PNG为JPG
        png_files = list(settings.DATA_RAW_UNFILTERED.glob("*.png"))
        for png_file in png_files:
            try:
                # 生成唯一的新文件名避免冲突
                jpg_path = png_file.with_suffix(".temp.jpg")
                img = cv2.imread(str(png_file))
                if img is None:
                    raise ValueError("无效的PNG文件")

                cv2.imwrite(
                    str(jpg_path),
                    img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                )
                png_file.unlink()
                logging.info(f"转换PNG到JPG: {png_file.name} -> {jpg_path.name}")
            except Exception as e:
                logging.error(f"PNG转换失败 {png_file.name}: {str(e)}")

        # 第二步：收集所有需要处理的文件
        all_files = []
        for f in settings.DATA_RAW_UNFILTERED.glob("*.*"):
            if f.suffix.lower() in (".jpg", ".jpeg", ".temp"):
                all_files.append(f)

        # 第三步：按创建时间+修改时间排序
        all_files.sort(key=lambda x: (x.stat().st_ctime, x.stat().st_mtime))

        # 第四步：创建临时目录存放所有文件
        temp_dir = settings.DATA_RAW_UNFILTERED / "temp_rename"
        temp_dir.mkdir(exist_ok=True)

        # 移动所有文件到临时目录
        moved_files = []
        for idx, file_path in enumerate(all_files):
            try:
                target = temp_dir / f"temp_{idx:08d}{file_path.suffix}"
                file_path.rename(target)
                moved_files.append(target)
            except Exception as e:
                logging.error(f"移动文件失败 {file_path.name}: {str(e)}")

        # 第五步：从临时目录重新编号
        for idx, src_path in enumerate(moved_files):
            try:
                # 转换为标准JPG格式
                img = cv2.imread(str(src_path))
                if img is None:
                    raise ValueError("无效的图片文件")

                # 生成最终路径
                target_path = settings.DATA_RAW_UNFILTERED / f"image_{idx}.jpg"

                # 保存并删除临时文件
                cv2.imwrite(
                    str(target_path),
                    img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                )
                src_path.unlink()
                logging.info(f"重新编号成功: {src_path.name} -> image_{idx}.jpg")
            except Exception as e:
                logging.error(f"处理失败 {src_path.name}: {str(e)}")

        # 清理临时目录
        try:
            temp_dir.rmdir()
        except:
            pass

        logging.info(f"共处理 {len(moved_files)} 张图片，最终编号: 0-{len(moved_files) - 1}")

    except Exception as e:
        logging.error(f"重命名流程异常: {str(e)}")
        raise