import os
import json
import cv2
import numpy as np
from pathlib import Path
from RoadConditionAI.configs.settings import DATA_RAW_DIR


def visualize_accident_detections():
    # 路径配置
    json_dir = Path("../data/processed/results/accident_rec")
    output_dir = Path("../data/processed/results/acc_vis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 可视化参数配置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    normal_color = (0, 255, 0)  # 绿色
    accident_color = (0, 0, 255)  # 红色
    text_thickness = 2
    box_thickness = 2

    # 遍历原始图像文件
    for img_file in DATA_RAW_DIR.glob("image_*.jpg"):
        # 解析文件序号
        try:
            base_name = img_file.stem  # 获取不带扩展名的文件名
            img_number = base_name.split("_")[-1]  # 提取序号
        except IndexError:
            print(f"文件名格式错误: {img_file}")
            continue

        # 构建对应JSON路径
        json_path = json_dir / f"image_{img_number}.json"
        if not json_path.exists():
            print(f"警告：{json_path} 不存在，跳过处理")
            continue

        # 读取并解析JSON数据
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"JSON解析失败: {json_path} - {str(e)}")
            continue

        # 加载原始图像
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"无法读取图像: {img_file}")
            continue

        # 遍历检测结果
        for vehicle in data.get("results", []):
            # 解析边界框坐标
            try:
                bbox = list(map(int, vehicle["bbox"]))  # 转换为整数
                x1, y1, x2, y2 = bbox
            except (KeyError, ValueError) as e:
                print(f"无效的bbox格式: {vehicle.get('bbox')}")
                continue

            # 确定绘制参数
            is_accident = vehicle.get("is_accident", False)
            vehicle_type = vehicle.get("vehicle_type", "unknown")
            color = accident_color if is_accident else normal_color

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

            # 构建标注文本
            text = f"{vehicle_type}"
            if is_accident:
                accident_type = vehicle.get("accident_type", "accident")
                text += f" - {accident_type}"

            # 计算文本位置（避免超出图像边界）
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
            text_y = max(y1 - 10, text_height + 5)  # 确保文本在框上方且不超出顶部

            # 绘制文本背景
            cv2.rectangle(img,
                        (x1, text_y - text_height - 5),
                        (x1 + text_width + 5, text_y + 5),
                        color,
                        -1)  # 填充矩形

            # 绘制文本
            cv2.putText(img,
                       text,
                       (x1 + 3, text_y - 5),
                       font,
                       font_scale,
                       (255, 255, 255),  # 白色文字
                       text_thickness,
                       cv2.LINE_AA)

        # 保存结果
        output_path = output_dir / f"vis_{base_name}.jpg"
        cv2.imwrite(str(output_path), img)
        print(f"可视化结果已保存至: {output_path}")

if __name__ == "__main__":
    visualize_accident_detections()
