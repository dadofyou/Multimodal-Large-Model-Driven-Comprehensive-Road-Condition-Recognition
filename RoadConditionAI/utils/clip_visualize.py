# visualizer.py
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import logging
from RoadConditionAI.configs.settings import (
    DATA_RAW_DIR,
    DATA_PROCESSED_VIS_ANN_DIR,  # CLIP json文件目录
    DATA_PROCESSED_VIS_ANN_IMG_DIR  # 可视化目录
)
import shutil

DATA_PROCESSED_VIS_ANN_IMG_DIR.mkdir(parents=True, exist_ok=True)

"""将CLIP筛选结果可视化"""

class RoadVisualizer:
    """道路分析可视化独立模块"""

    def __init__(self, config=None):
        self.config = config or {
            "colors": {
                "car": "#FF0000",  # 红
                "crack": "#808080",  # 灰
                "litter": "#00FF00",  # 绿
                "lines": "#FFA500",  # 橙
                "unknown": "#0000FF"  # 蓝
            },
            "font_size": 14,
            "opacity": 0.5
        }

        # 字体加载优化
        try:
            self.font = ImageFont.truetype("Arial Unicode.ttf", self.config["font_size"])  # 支持中文
        except:
            try:
                self.font = ImageFont.truetype("NotoSansCJK-Regular.ttc", self.config["font_size"])
            except:
                self.font = ImageFont.load_default(size=self.config["font_size"])
                logging.warning("使用默认字体")

    def _clean_output_dir(self, output_dir: Path):
        """安全清空输出目录"""
        try:
            if output_dir.exists():
                # 递归删除目录内容（网页1/网页4方案）
                shutil.rmtree(output_dir, ignore_errors=True)
                # 延迟确保删除完成（针对Windows系统）
                import time
                time.sleep(0.5)
            # 重建目录结构（网页2方案）
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"清空目录失败: {str(e)}")
            raise
    def _validate_bbox(self, img, bbox):
        """校正边界框坐标"""
        x1, y1, x2, y2 = (
            max(0, min(bbox[0], img.width)),
            max(0, min(bbox[1], img.height)),
            max(0, min(bbox[2], img.width)),
            max(0, min(bbox[3], img.height))
        )
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def generate_visualization(self, image_path: Path, json_path: Path,
                               output_dir: Path = None, mode: str = "basic"):
        """
        生成可视化结果

        参数：
        - image_path: 原始图片路径
        - json_path: CLIP结果JSON路径
        - output_dir: 输出目录（默认同源目录）
        - mode: 可视化模式（basic/mask/heatmap）
        """
        try:
            # 加载数据
            raw_image = Image.open(image_path)
            with open(json_path, 'r') as f:
                data = json.load(f)

            # 创建输出路径
            output_dir = output_dir or image_path.parent
            output_path = output_dir / f"vis_{image_path.name}"

            # 选择可视化模式
            if mode == "basic":
                self._draw_basic(raw_image, data["regions"])
            elif mode == "mask":
                self._draw_with_mask(raw_image, data["regions"])
            elif mode == "heatmap":
                self._draw_heatmap(raw_image, data["regions"])
            else:
                raise ValueError(f"未知模式：{mode}")

            raw_image.save(output_path)
            logging.info(f"可视化结果保存至：{output_path}")
            return output_path

        except Exception as e:
            logging.error(f"可视化失败：{str(e)}")
            return None

    def _draw_basic(self, image, regions):
        """基础框体绘制"""
        draw = ImageDraw.Draw(image)
        for region in regions:
            bbox = self._validate_bbox(image, region["bbox"])
            if not bbox:
                continue

            color = self.config["colors"].get(region["category"], "#000000")
            # 绘制半透明矩形
            draw.rectangle(bbox,
                           outline=color,
                           width=2)
            # 添加标签
            label = f"{region['category']} ({region['clip_score']:.2f})"
            text_x = bbox[0] + 5 if (bbox[0] + 50 < image.width) else bbox[0] - 50
            text_y = bbox[1] + 5 if (bbox[1] + 20 < image.height) else bbox[1] - 20
            draw.text((text_x, text_y), label, fill=color, font=self.font)

    def _draw_with_mask(self, image, regions):
        """掩膜模式（需要JSON中包含掩膜数据）"""
        # 实现细节类似基础模式，需处理多边形绘制
        pass

    def _draw_heatmap(self, image, regions):
        """置信度热力图模式"""
        # 实现细节参考之前的方案
        pass

    # 批量识别
    def process_mult_images(self):
        # 预处理：清空可视化目录
        self._clean_output_dir(DATA_PROCESSED_VIS_ANN_IMG_DIR)
        count = 0
        for img_path in DATA_RAW_DIR.glob("*.jpg"):
            result = visualizer.generate_visualization(
                img_path,
                Path(DATA_PROCESSED_VIS_ANN_DIR / f"{img_path.stem}.json"),
                Path(DATA_PROCESSED_VIS_ANN_IMG_DIR)
            )
            count += 1
            print(f"生成结果：{result}" if result else "生成失败")
        return count


if __name__ == "__main__":
    visualizer = RoadVisualizer()
    num = visualizer.process_mult_images()
    print(f"共生成{num}张结果")

