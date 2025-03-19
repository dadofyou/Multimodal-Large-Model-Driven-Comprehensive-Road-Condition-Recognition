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
    handlers=[logging.StreamHandler()]
)


def process_images():
    """更新后的处理流程"""
    try:
        segmenter = RoadSegmenter()
        image_files = sorted(settings.DATA_RAW_DIR.glob("image_*.jpg"))

        with tqdm(image_files, unit="img") as pbar:
            for img_path in pbar:
                result = segmenter.process_image(str(img_path))
                if not result:
                    continue

                # 保存JSON到原目录
                json_path = settings.DATA_PROCESSED_DIR / f"{img_path.stem}_sub.json"
                with open(json_path, 'w') as f:
                    json.dump({
                        "original_size": result["original_size"],
                        "sub_num": result["sub_num"],
                        "sub_regions": result["sub_regions"]
                    }, f, indent=2)

                # 保存可视化到新目录
                vis_path = settings.DATA_PROCESSED_VIS_DIR / f"{img_path.stem}_vis.jpg"  # 修改路径
                cv2.imwrite(str(vis_path), result["visualization"])

                pbar.set_postfix_str(f"Regions: {result['sub_num']}")

    except Exception as e:
        logging.error(f"处理失败: {str(e)}")
        raise


if __name__ == "__main__":
    process_images()