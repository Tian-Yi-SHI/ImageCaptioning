"""
数据批处理函数
处理变长序列的batch，进行padding和打包
"""
import torch
from typing import List, Tuple, Any

from .vocabulary import Vocabulary


def collate_fn(batch: List[Tuple[Any, List[str], str]], 
                vocabulary: Vocabulary, 
                max_caption_length: int = 30) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    将batch数据整理成模型可用的格式
    
    Args:
        batch: DataLoader返回的batch列表，每个元素是(image, captions, image_name)
        vocabulary: 词汇表对象
        max_caption_length: caption的最大长度（包括START和END）
        
    Returns:
        images: (batch_size, 3, H, W) 图像tensor
        captions: (batch_size, max_length) caption索引tensor
        caption_lengths: (batch_size,) 每个caption的实际长度（不包括padding）
        image_names: batch中图像名称列表
    """
    images = []
    captions_list = []
    caption_lengths = []
    image_names = []
    
    # 从每个样本中选择第一个caption（训练时可以用不同的caption）
    for image, captions, image_name in batch:
        images.append(image)
        image_names.append(image_name)
        
        # 使用第一个caption
        caption = captions[0]
        caption_indices = vocabulary.caption_to_indices(caption, max_length=max_caption_length)
        captions_list.append(caption_indices)
        
        # 计算实际长度（不包括padding）
        # 通过caption_to_indices获取，然后计算非PAD的长度
        caption_indices_full = vocabulary.caption_to_indices(caption, max_length=None)
        actual_length = len(caption_indices_full)
        actual_length = min(actual_length, max_caption_length)
        caption_lengths.append(actual_length)
    
    # 转换为tensor
    images = torch.stack(images, dim=0)
    captions = torch.tensor(captions_list, dtype=torch.long)
    caption_lengths = torch.tensor(caption_lengths, dtype=torch.long)
    
    return images, captions, caption_lengths, image_names

