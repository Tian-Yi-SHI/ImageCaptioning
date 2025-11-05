"""
词汇表构建和管理模块
用于从数据集中构建词汇表，并提供词汇与索引的转换功能
"""
from collections import Counter

import re
from typing import List, Dict, Optional


class Vocabulary:
    """
    词汇表类，管理词汇与索引的映射关系
    """
    
    def __init__(self, max_vocab_size: int = 5000):
        """
        初始化词汇表
        
        Args:
            max_vocab_size: 最大词汇量（不包括特殊token）
        """
        self.max_vocab_size = max_vocab_size
        
        # 特殊token
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        
        # 词汇到索引的映射
        self.word2idx: Dict[str, int] = {}
        # 索引到词汇的映射
        self.idx2word: Dict[int, str] = {}
        
        # 是否已构建
        self.is_built = False
    
    def _tokenize(self, caption: str) -> List[str]:
        """
        将caption分词（简单版本，按空格和标点分词）
        
        Args:
            caption: 原始caption字符串
            
        Returns:
            分词后的词汇列表
        """
        # 转换为小写
        caption = caption.lower()
        # 移除多余空格
        caption = re.sub(r'\s+', ' ', caption)
        # 简单分词：按空格和常见标点分割
        tokens = re.findall(r'\b\w+\b', caption)
        return tokens
    
    def tokenize(self, caption: str) -> List[str]:
        """
        公共方法：将caption分词
        
        Args:
            caption: 原始caption字符串
            
        Returns:
            分词后的词汇列表
        """
        return self._tokenize(caption)
    
    def build_vocabulary(self, captions_list: List[List[str]], min_word_freq: int = 5, auto_adjust: bool = True):
        """
        从captions列表中构建词汇表（按照《Show and Tell》论文方法）
        
        Args:
            captions_list: 所有captions的列表，每个元素是某个图像的多个caption列表
            min_word_freq: 最小词频阈值（论文：至少5次）
            auto_adjust: 是否自动调整max_vocab_size以适应实际词汇数量（论文方法）
        """
        # 统计所有词汇的频率
        word_counter = Counter()
        
        for captions in captions_list:
            for caption in captions:
                tokens = self._tokenize(caption)
                word_counter.update(tokens)
        
        print(f"  总不同词汇数: {len(word_counter)}")
        
        # 添加特殊token
        self.word2idx[self.PAD_TOKEN] = 0
        self.word2idx[self.START_TOKEN] = 1
        self.word2idx[self.END_TOKEN] = 2
        self.word2idx[self.UNK_TOKEN] = 3
        
        # 添加最常见的词汇（按频率排序）
        # 论文：仅保留出现至少min_word_freq次的单词
        filtered_words = [(word, count) for word, count in word_counter.most_common() 
                         if count >= min_word_freq]
        
        # 按照论文方法：根据训练集的实际情况自动调整词汇表大小
        actual_vocab_size = len(filtered_words)
        
        if auto_adjust:
            # 自动调整max_vocab_size以适应实际词汇数量（向上取整到最近的1000）
            # 例如：2872个词汇 -> 调整为3000
            adjusted_size = ((actual_vocab_size // 1000) + 1) * 1000
            self.max_vocab_size = adjusted_size
            print(f"  自动调整max_vocab_size: {self.max_vocab_size} (实际词汇数: {actual_vocab_size})")
        
        # 限制词汇表大小（如果设置了max_vocab_size）
        if len(filtered_words) > self.max_vocab_size:
            filtered_words = filtered_words[:self.max_vocab_size]
            print(f"  警告: 词汇数量({len(filtered_words)})超过max_vocab_size({self.max_vocab_size})，已截断")
        
        for idx, (word, count) in enumerate(filtered_words, start=4):
            self.word2idx[word] = idx
        
        # 构建反向映射
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        self.is_built = True
        
        # 计算词汇覆盖率：过滤后的词汇在总词频中的占比（论文方法验证）
        total_word_count = sum(word_counter.values())
        filtered_word_count = sum(count for _, count in filtered_words)
        coverage = (filtered_word_count / total_word_count * 100) if total_word_count > 0 else 0
        
        # 输出统计信息（按照论文方法）
        print(f"词汇表构建完成，共 {len(self.word2idx)} 个词汇")
        print(f"  - 特殊token: 4个 (PAD, START, END, UNK)")
        print(f"  - 普通词汇: {len(filtered_words)}个 (出现至少{min_word_freq}次)")
        print(f"  - 词汇覆盖率: {coverage:.2f}% (过滤后词汇占总词频的比例)")
    
    def caption_to_indices(self, caption: str, max_length: Optional[int] = None) -> List[int]:
        """
        将caption转换为索引序列
        
        Args:
            caption: 原始caption字符串
            max_length: 最大长度（如果超过会截断，不足会padding）
            
        Returns:
            索引列表，包含START和END token
        """
        if not self.is_built:
            raise RuntimeError("词汇表尚未构建，请先调用build_vocabulary")
        
        tokens = self._tokenize(caption)
        
        # 转换为索引
        indices = [self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]) for token in tokens]
        
        # 添加START和END
        indices = [self.word2idx[self.START_TOKEN]] + indices + [self.word2idx[self.END_TOKEN]]
        
        # 处理长度
        if max_length is not None:
            if len(indices) > max_length:
                # 截断：保留START，截断到max_length-1，最后加END
                indices = [indices[0]] + indices[1:max_length-1] + [self.word2idx[self.END_TOKEN]]
            elif len(indices) < max_length:
                # Padding
                indices = indices + [self.word2idx[self.PAD_TOKEN]] * (max_length - len(indices))
        
        return indices
    
    def indices_to_caption(self, indices: List[int], remove_special: bool = True) -> str:
        """
        将索引序列转换为caption字符串
        
        Args:
            indices: 索引列表
            remove_special: 是否移除特殊token（PAD, START, END）
            
        Returns:
            caption字符串
        """
        if not self.is_built:
            raise RuntimeError("词汇表尚未构建，请先调用build_vocabulary")
        
        words = []
        for idx in indices:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if remove_special:
                    if word not in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]:
                        words.append(word)
                else:
                    words.append(word)
        
        return ' '.join(words)
    
    def __len__(self):
        """返回词汇表大小"""
        return len(self.word2idx)
    
    def get_pad_idx(self) -> int:
        """获取PAD token的索引"""
        return self.word2idx[self.PAD_TOKEN]
    
    def get_start_idx(self) -> int:
        """获取START token的索引"""
        return self.word2idx[self.START_TOKEN]
    
    def get_end_idx(self) -> int:
        """获取END token的索引"""
        return self.word2idx[self.END_TOKEN]
    
    def get_unk_idx(self) -> int:
        """获取UNK token的索引"""
        return self.word2idx[self.UNK_TOKEN]

