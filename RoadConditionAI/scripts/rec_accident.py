import logging
from datetime import datetime
from pathlib import Path
import json
import glob
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm

# 输入目录
DATA_RAW_DIR = "../data/processed/results"


class CLIPRecognizer:
    def __init__(self, model_name="ViT-B/32", device=None):
        # 设备自动选择
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.model = self.model.float()  # 新增此行

        # 预定义车辆属性分类体系
        self.text_prompts = {
            "accident": [
                "normal traffic scene with vehicles moving orderly",  # 更详细的正常描述
                "traffic accident scene showing visible damage, or vehicle overturned"
            ],
            "accident_type": [
                # 车辆损坏
                "overturned vehicle on the center of the road or lying on its side at over 90 degree angle",

                # 车辆侧翻
                "vehicle with visible damage on its surface or car front end completely crushed"
            ]
        }

    def encode_image(self, image_path):
        # 增加多尺度特征融合
        image = Image.open(image_path).convert("RGB")
        multi_scale_features = []
        for scale in [1.0, 0.8, 1.2]:  # 多尺度处理
            scaled_img = image.resize((int(image.width*scale), int(image.height*scale)))
            img_input = self.preprocess(scaled_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(img_input)
                features /= features.norm(dim=-1, keepdim=True)
            multi_scale_features.append(features)
        return torch.mean(torch.stack(multi_scale_features), dim=0).squeeze(0)

    def _calculate_similarity(self, feature, labels):
        text_inputs = torch.cat([clip.tokenize(l) for l in labels]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return (feature @ text_features.T).softmax(dim=-1).cpu().numpy()

    def predict_attributes(self, image_feature, accident_threshold=0.501):
        """多属性联合预测"""
        """返回结构化预测结果"""
        result = {
            "has_accident": False,
            "accident_type": None
        }

        accident_probs = self._calculate_similarity(image_feature, self.text_prompts["accident"])
        accident_confidence = accident_probs[1]  # 事故类别的概率

        result["has_accident"] = bool(accident_confidence > accident_threshold)
        result["accident_confidence"] = float(accident_confidence)  # 保留置信度用于调试
        print(
            f"[Threshold Check] Current threshold: {accident_threshold},"
            f" Confidence: {accident_confidence:.4f},"
            f" Decision: {result['has_accident']}")

        # 若检测到事故则分类事故类型
        if result["has_accident"]:
            type_labels = self.text_prompts["accident_type"]
            type_probs = self._calculate_similarity(image_feature, type_labels)
            result["accident_type"] = type_labels[np.argmax(type_probs)]
        return result


def main():
    # 初始化组件
    recognizer = CLIPRecognizer()
    base_path = Path(DATA_RAW_DIR)

    # 输入配置
    input_dirs = sorted(glob.glob(str(base_path / "yolo_img_division_vis" / "image_*")))
    # 遍历处理每个输入目录
    for input_dir in tqdm(input_dirs, desc="Processing directories"):
        input_dir = Path(input_dir)

        # 生成输出路径
        output_dir = base_path / "accident_rec"
        output_path = output_dir / f"{input_dir.name}.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取当前目录的图片列表
        image_paths = sorted(glob.glob(str(input_dir / "vehicle_*.jpg")))
        json_path = base_path / "yolo_img_division" / f"{ input_dir.name }_result.json"
        # 检查 JSON 文件是否存在
        if not json_path.exists():
            print(f"错误：未找到 JSON 文件 {json_path}")
            continue
        results = []
        # 处理当前目录的图片
        results = []
        try:
            # 读取 JSON 文件
            with open(json_path, 'r') as f:
                data = json.load(f)
            # 遍历 JSON 中的每辆车
            for vehicle in data["vehicles"]:
                # 获取裁剪图片路径（例如 image_0/vehicle_0.jpg）
                crop_img_path = base_path / "yolo_img_division_vis" / vehicle["crop_image_path"]

                # 检查裁剪图片是否存在
                if not crop_img_path.exists():
                    print(f"警告：裁剪图片 {crop_img_path} 不存在")
                    continue
                # 特征提取和属性预测
                features = recognizer.encode_image(str(crop_img_path))
                attributes = recognizer.predict_attributes(features)
                # 构建每辆车的独立记录
                record = {
                    "vehicle_id": vehicle["vehicle_id"],
                    "original_image": input_dir.name,  # 例如 image_0.jpg 对应的目录名
                    "crop_image": str(vehicle["crop_image_path"]),
                    "vehicle_type": vehicle["class_name"],
                    "is_accident": attributes["has_accident"],
                    "bbox": vehicle["original_bbox"],
                }
                if attributes["has_accident"]:
                    record["accident_type"] = attributes["accident_type"]
                results.append(record)
        except Exception as e:
            print(f"处理 {json_path} 失败: {str(e)}")
            continue
            # 保存结果
        output_data = {
            "metadata": {
                "source_directory": str(input_dir.relative_to(base_path)),
                "processing_time": datetime.now().isoformat(),
                "total_vehicles": len(results)
            },
            "results": results
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"成功处理 {input_dir.name}: {len(results)}/{len(image_paths)} 图片已保存到 {output_path}")


if __name__ == "__main__":
    main()
    from RoadConditionAI.utils.accident_vis import visualize_accident_detections
    visualize_accident_detections()


