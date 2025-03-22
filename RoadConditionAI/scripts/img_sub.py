# RoadConditionAI/scripts/img_sub.py
import json
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
from ..configs import settings
from ..models.detector import RoadSegmenter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('processing.log')]
)
logger = logging.getLogger(__name__)

def process_images():
    """顺序处理图像，避免并发带来的线程安全问题"""
    try:
        segmenter = RoadSegmenter()
        logger.info(f"初始化完成，使用设备: {settings.DEVICE}")
        image_files = sorted(settings.DATA_RAW_DIR.glob("image_*.jpg"))
        if not image_files:
            logger.warning("未找到输入图像")
            return

        for img_path in tqdm(image_files, desc="处理进度", unit="img"):
            result = segmenter.process_image(str(img_path))
            if result:
                save_results(img_path, result)
                tqdm.write(f"{img_path.name} 处理完成，Regions: {result['sub_num']}")
            else:
                logger.error(f"处理失败 {img_path.name}")
    except Exception as e:
        logger.error(f"致命错误: {str(e)}", exc_info=True)
        raise

def save_results(img_path: Path, result: dict):
    """保存JSON与可视化图像"""
    try:
        json_path = settings.DATA_PROCESSED_DIR / f"{img_path.stem}_sub.json"
        with open(json_path, 'w') as f:
            json.dump({
                "original_size": result["original_size"],
                "sub_num": result["sub_num"],
                "sub_regions": result["sub_regions"]
            }, f, indent=2)
        vis_path = settings.DATA_PROCESSED_VIS_DIR / f"{img_path.stem}_vis.jpg"
        cv2.imencode('.jpg', result["visualization"],
                     [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(str(vis_path))
    except Exception as e:
        logger.error(f"保存失败 {img_path.name}: {str(e)}")

if __name__ == "__main__":
    settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    settings.DATA_PROCESSED_VIS_DIR.mkdir(parents=True, exist_ok=True)
    process_images()
