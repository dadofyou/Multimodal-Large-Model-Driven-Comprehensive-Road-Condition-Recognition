# RoadConditionAI/main.py
import sys
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from RoadConditionAI.scripts.img_sub import process_images

if __name__ == "__main__":
    print("启动道路状况分析系统...")
    process_images()
    print("处理完成！")