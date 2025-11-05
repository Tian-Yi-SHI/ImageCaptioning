"""
未来迭代计划：图像caption可视化扩展模块

此模块预留了后续扩展可视化功能的接口和示例代码结构。
当前阶段（Month 1）仅实现文本输出，后续可以在此模块中添加：

1. 图像+文本对比可视化
2. 注意力热力图可视化
3. 批量结果可视化

使用方法：
    # 当前阶段：仅使用文本输出（已在PipelineIC中实现）
    
    # 后续扩展（Month 2+）：
    # from utils.future_visualization import visualize_caption_results
    # visualize_caption_results(images, captions, attention_weights)
"""

import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Optional

# 设置中文字体（Mac适配）
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def visualize_caption_comparison(image_paths: List[str],
                                 true_captions: List[str],
                                 generated_captions: List[str],
                                 output_dir: str = "./storage/log/visualizations",
                                 num_samples: int = 10):
    """
    可视化caption对比结果（图像+文本）
    
    此函数将在后续迭代中实现，用于：
    - 显示图像
    - 对比真实caption和生成caption
    - 保存对比图到文件
    
    Args:
        image_paths: 图像路径列表
        true_captions: 真实caption列表
        generated_captions: 生成caption列表
        output_dir: 输出目录
        num_samples: 要可视化的样本数量
    """
    # TODO: Month 2+ 实现
    # 1. 创建输出目录
    # 2. 为每个样本创建对比图
    # 3. 使用matplotlib显示图像和caption
    # 4. 保存图像
    pass


def visualize_attention_heatmap(image_path: str,
                                caption: str,
                                attention_weights: torch.Tensor,
                                output_path: Optional[str] = None):
    """
    可视化注意力热力图
    
    此函数将在后续迭代中实现（需要模型支持注意力机制），用于：
    - 显示图像
    - 在每个词上叠加注意力热力图
    - 展示模型关注的重点区域
    
    Args:
        image_path: 图像路径
        caption: 生成的caption
        attention_weights: 注意力权重 (seq_len, H, W)
        output_path: 输出路径（可选）
    """
    # TODO: Month 2+ 实现（需要注意力机制）
    # 1. 加载图像
    # 2. 为每个词生成注意力热力图
    # 3. 叠加显示
    # 4. 保存结果
    pass


def create_batch_visualization(results: Dict[str, Dict],
                                output_dir: str = "./storage/log/visualizations"):
    """
    批量可视化结果
    
    创建包含多个样本的网格可视化
    
    Args:
        results: 结果字典，格式 {image_name: {true_caption: ..., generated_caption: ..., ...}}
        output_dir: 输出目录
    """
    # TODO: Month 2+ 实现
    # 1. 创建网格布局
    # 2. 为每个样本创建子图
    # 3. 显示图像和caption
    # 4. 保存整体可视化
    pass


def export_caption_results_to_html(results: List[Dict],
                                   output_path: str = "./storage/log/visualizations/results.html"):
    """
    将caption结果导出为HTML文件（便于在浏览器中查看）
    
    Args:
        results: 结果列表，每个元素包含 {image_name, true_caption, generated_caption, ...}
        output_path: 输出HTML文件路径
    """
    # TODO: Month 2+ 实现
    # 1. 创建HTML模板
    # 2. 嵌入图像（base64编码或相对路径）
    # 3. 显示caption对比
    # 4. 保存HTML文件
    pass


