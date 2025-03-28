import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import logging
from RoadConditionAI.configs.settings import (
    DATA_PROCESSED_VIS_ANN_DIR,  # CLIP JSON 文件目录
    DATA_PROCESSED_VIS_ANN_IMG_DIR  # 可视化输出目录
)
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def visualize_detection(image_path: Path, json_path: Path, output_dir: Path = None, font_size: int = 24):
    """
    生成可视化图像：读取检测 JSON 文件，在原始图像上绘制标识框（统一使用红色）和标签文字，
    标签文字大小可通过 font_size 参数自定义，并保存到指定输出目录中。

    参数：
      image_path: 原始图片路径
      json_path: CLIP 检测结果 JSON 文件路径
      output_dir: 输出目录（默认为 DATA_PROCESSED_VIS_ANN_IMG_DIR）
      font_size: 标签文字的字体大小（默认 24）

    返回：
      成功则返回生成的图像路径，否则返回 None
    """
    try:
        raw_image = Image.open(image_path)
    except Exception as e:
        logger.error(f"无法打开图像 {image_path.name}: {e}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"无法加载 JSON 文件 {json_path.name}: {e}")
        return None

    # 固定使用红色 (#FF0000) 绘制标识框和标签
    red_color = "#FF0000"
    draw = ImageDraw.Draw(raw_image)

    # 尝试加载字体，优先使用 "DejaVuSans.ttf"，其次 "Arial.ttf"，均失败则使用默认字体
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception as e1:
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except Exception as e2:
            logger.warning(f"无法加载指定字体，使用默认字体。错误信息: {e1}; {e2}")
            font = ImageFont.load_default()

    regions = data.get("regions", [])
    for region in regions:
        bbox = region.get("bbox", [])
        if len(bbox) != 4:
            continue
        # 绘制边界框
        draw.rectangle(bbox, outline=red_color, width=2)
        # 构造标签文本：显示所有类别及其置信度
        labels = [f"{cat}: {score:.2f}" for cat, score in
                  zip(region.get("categories", []), region.get("confidence_scores", []))]
        label_text = "\n".join(labels)
        # 默认在边界框左上角偏移 5 个像素处绘制文本
        text_x = bbox[0] + 5
        text_y = bbox[1] + 5
        draw.multiline_text((text_x, text_y), label_text, fill=red_color, font=font)

    output_dir = output_dir or DATA_PROCESSED_VIS_ANN_IMG_DIR
    output_path = output_dir / f"vis_{image_path.name}"
    try:
        raw_image.save(output_path)
        logger.info(f"可视化结果保存至：{output_path}")
        return output_path
    except Exception as e:
        logger.error(f"保存可视化图像失败 {output_path.name}: {e}")
        return None
