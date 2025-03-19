# utils/platform_adapter.py
import platform
import os
import torch
import RoadConditionAI.configs.settings as settings


def setup_runtime():
    """自动适配运行环境"""
    system = platform.system()

    if system == "Windows":
        # Windows特定配置
        torch.set_num_threads(os.cpu_count() // 2)  # 避免内存溢出
    elif system == "Linux":
        # Linux优化配置
        torch.backends.cudnn.benchmark = True  # 启用加速算法

    if settings.DEVICE == "cpu":
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())