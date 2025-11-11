"""
图像描述生成Pipeline实现
包含训练和测试的完整逻辑
"""
import os
import time
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .Pipeline import Pipeline
from models.ImageCaptionModel import ImageCaptionModel
from utils.vocabulary import Vocabulary
from utils.metrics import calculate_bleu_scores
from config.config import define_dev


class PipelineIC(Pipeline):
    """
    图像描述生成Pipeline的具体实现
    """
    
    def __init__(self, vocabulary: Optional[Vocabulary] = None, device: Optional[torch.device] = None):
        """
        初始化Pipeline
        
        Args:
            vocabulary: 词汇表对象（如果为None，需要后续构建）
            device: 计算设备（如果为None，自动检测）
        """
        super().__init__()
        
        self.vocabulary = vocabulary
        self.device = device if device is not None else define_dev()
        self.max_caption_length = 25  # 从30减少到25，加快训练
        
        # 训练历史记录
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def build_model(self, model: Optional[Any] = None, 
                    vocab_size: Optional[int] = None,
                    embed_dim: int = 256,
                    hidden_dim: int = 512,
                    num_layers: int = 1,
                    dropout: float = 0.5,
                    freeze_encoder: bool = False) -> Any:
        """
        构建或加载模型
        
        Args:
            model: 预训练模型（可选）
            vocab_size: 词汇表大小（如果vocabulary已设置，会自动获取）
            embed_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout概率
            
        Returns:
            构建的模型
        """
        if model is not None:
            # 加载已有模型
            self.model = model
        else:
            # 构建新模型
            if vocab_size is None:
                if self.vocabulary is None:
                    raise ValueError("vocabulary未设置，无法确定vocab_size")
                vocab_size = len(self.vocabulary)
            
            self.model = ImageCaptionModel(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                max_caption_length=self.max_caption_length,
                dropout=dropout,
                freeze_encoder=freeze_encoder
            )
        
        # 将模型移到设备
        self.model = self.model.to(self.device)
        
        # 标记模型已准备
        self._mark_model_ready()
        
        print(f"模型构建完成，已移动到设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def set_vocabulary(self, vocabulary: Vocabulary):
        """设置词汇表"""
        self.vocabulary = vocabulary
        self.max_caption_length = 30
    
    def _train_core(self, train_loader: Any, val_loader: Optional[Any] = None, 
                    checkpoint_path: Optional[str] = None, checkpoint_dir: Optional[str] = None) -> dict:
        """
        训练核心逻辑
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            checkpoint_path: checkpoint文件路径（如果提供，则从该checkpoint恢复训练）
            checkpoint_dir: checkpoint保存目录（如果提供，每5轮保存一次checkpoint）
            
        Returns:
            训练结果字典
        """
        if self.vocabulary is None:
            raise RuntimeError("词汇表未设置，请先设置vocabulary")
        
        # 获取训练参数
        batch_size = self.training_params['batch_size']
        n_epoch = self.training_params['n_epoch']
        lr = self.training_params['lr']
        wd = self.training_params['wd']
        
        # 设置优化器（按照《Show and Tell》论文）
        # 论文：SGD without momentum，但实际中Adam通常更好
        # 这里提供两种选择：SGD（论文配置）或Adam（推荐）
        use_sgd = False  # 设置为True使用SGD（论文配置），False使用Adam（推荐）
        
        if use_sgd:
            # 论文配置：SGD without momentum
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0, weight_decay=wd)
            # 论文：固定学习率
            scheduler = None
            print("使用SGD优化器（论文配置）")
        else:
            # 推荐配置：Adam（通常效果更好）
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
            # 学习率调度器（注意：verbose参数在新版PyTorch中已移除）
            # factor=0.8: 每次降低20%（更温和），patience=3: 验证损失3个epoch不降才降低学习率
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-5
            )
            print("使用Adam优化器（推荐，效果通常更好）")
        
        # 损失函数（忽略PAD token）
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocabulary.get_pad_idx())
        
        # 训练历史
        best_val_loss = float('inf')
        start_epoch = 1
        
        # 如果提供了checkpoint路径，尝试加载checkpoint恢复训练
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n从checkpoint恢复训练: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 恢复模型状态
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("  ✓ 模型状态已恢复")
            
            # 恢复优化器状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("  ✓ 优化器状态已恢复")
            
            # 恢复scheduler状态
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("  ✓ 学习率调度器状态已恢复")
            
            # 恢复训练历史
            if 'train_history' in checkpoint:
                self.train_history = checkpoint['train_history']
                print("  ✓ 训练历史已恢复")
            
            # 恢复epoch和最佳验证损失
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"  ✓ 从第 {start_epoch} 轮继续训练")
            
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                print(f"  ✓ 当前最佳验证损失: {best_val_loss:.4f}")
        
        start_time = time.time()
        
        print(f"\n开始训练，共 {n_epoch} 个epoch（从第 {start_epoch} 轮开始）")
        print(f"训练集大小: {len(train_loader.dataset)}")
        if val_loader:
            print(f"验证集大小: {len(val_loader.dataset)}")
        if checkpoint_dir:
            print(f"Checkpoint保存目录: {checkpoint_dir}")
            print(f"每5轮保存一次checkpoint")
        print("-" * 60)
        
        for epoch in range(start_epoch, n_epoch + 1):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epoch} [Train]")
            for images, captions, caption_lengths, image_names in train_pbar:
                # 移到设备
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(images, captions, caption_lengths)
                
                # 计算损失（outputs: [batch, seq_len-1, vocab_size], targets: [batch, seq_len-1]）
                targets = captions[:, 1:]  # 去掉START token
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                # 梯度裁剪：提高max_norm到5.0，允许更大的梯度
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                
                # 记录损失
                train_loss += loss.item()
                train_batches += 1
                
                # 更新进度条
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / train_batches
            self.train_history['train_loss'].append(avg_train_loss)
            self.train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # 验证阶段（每2个epoch验证一次，加快训练）
            val_loss = None
            if val_loader and (epoch % 2 == 0 or epoch == 1 or epoch == n_epoch):
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    # 使用disable=False但减少详细输出
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{n_epoch} [Val]", leave=False)
                    for images, captions, caption_lengths, image_names in val_pbar:
                        images = images.to(self.device)
                        captions = captions.to(self.device)
                        
                        outputs = self.model(images, captions, caption_lengths)
                        targets = captions[:, 1:]
                        outputs = outputs.reshape(-1, outputs.size(-1))
                        targets = targets.reshape(-1)
                        
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        val_batches += 1
                        
                        val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                avg_val_loss = val_loss / val_batches
                self.train_history['val_loss'].append(avg_val_loss)
                
                # 学习率调度（如果有scheduler）
                if scheduler is not None:
                    scheduler.step(avg_val_loss)
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"  ✓ 验证损失改善，保存最佳模型 (val_loss: {avg_val_loss:.4f})")
            else:
                # 如果没有验证，使用训练损失作为调度器输入（避免报错）
                if val_loader and scheduler is not None:
                    scheduler.step(avg_train_loss)
            
            # 每5轮保存一次checkpoint（如果提供了checkpoint_dir）
            if checkpoint_dir and epoch % 5 == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                self.save_checkpoint(
                    checkpoint_file, 
                    epoch, 
                    optimizer, 
                    scheduler, 
                    best_val_loss if val_loader else None
                )
                print(f"  ✓ 已保存checkpoint: {checkpoint_file}")
            
            # 打印epoch总结
            print(f"\nEpoch {epoch}/{n_epoch} 完成:")
            print(f"  训练损失: {avg_train_loss:.4f}")
            if val_loss is not None:
                print(f"  验证损失: {avg_val_loss:.4f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 60)
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {training_time/60:.2f} 分钟")
        
        return {
            'train_history': self.train_history,
            'training_time': training_time,
            'best_val_loss': best_val_loss if val_loader else None
        }
    
    def _test_core(self, test_loader: Any) -> dict:
        """
        测试核心逻辑
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            测试结果字典
        """
        if self.vocabulary is None:
            raise RuntimeError("词汇表未设置，请先设置vocabulary")
        
        # 损失函数
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocabulary.get_pad_idx())
        
        # 测试模式
        self.model.eval()
        
        test_loss = 0.0
        test_batches = 0
        
        # 从dataset中获取所有参考caption（用于BLEU评估）
        # 注意：这里假设dataset返回的是(image, captions, image_name)格式
        # 其中captions是字符串列表
        print("\n正在收集参考caption...")
        dataset = test_loader.dataset
        
        # 如果dataset是Subset，需要获取原始dataset
        # 可能需要多层解包：Subset -> TransformDatasetWrapper -> Subset -> Flickr8KDataset
        original_dataset = dataset
        indices = list(range(len(dataset)))
        
        # 解包Subset和TransformDatasetWrapper，找到原始Flickr8KDataset
        while hasattr(original_dataset, 'dataset'):
            if hasattr(original_dataset, 'indices'):
                # 如果是Subset，需要获取索引
                indices = [original_dataset.indices[i] for i in indices]
            original_dataset = original_dataset.dataset
        
        # 构建image_name到所有参考caption的映射
        references_raw = {}  # {image_name: [list of raw caption strings]}
        for idx in indices:
            try:
                # 优先避免触发图像加载
                if hasattr(original_dataset, 'image_names') and hasattr(original_dataset, 'image_captions'):
                    image_name = original_dataset.image_names[idx]
                    captions = original_dataset.image_captions.get(image_name, [])
                elif hasattr(original_dataset, 'get_captions_by_index'):
                    captions, image_name = original_dataset.get_captions_by_index(idx)
                else:
                    # 回退到__getitem__，可能会加载图像
                    _, captions, image_name = original_dataset[idx]
                if image_name not in references_raw:
                    references_raw[image_name] = []
                if isinstance(captions, list):
                    references_raw[image_name].extend(captions)
                else:
                    references_raw[image_name].append(captions)
            except Exception as e:
                # 如果无法获取，跳过
                print(f"  警告: 无法获取索引 {idx} 的参考caption: {e}")
                continue
        
        print(f"已收集 {len(references_raw)} 个图像的参考caption")
        
        # 用于BLEU评估（转换为词汇列表格式）
        references = {}  # {image_name: [list of reference caption word lists]}
        hypotheses = {}  # {image_name: generated caption word list}
        
        # 用于示例输出
        example_samples = []  # 存储前几个样本
        
        print(f"\n开始测试，测试集大小: {len(test_loader.dataset)}")
        print("-" * 60)
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="测试中")
            for batch_idx, (images, captions, caption_lengths, image_names) in enumerate(test_pbar):
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                # 计算损失
                outputs = self.model(images, captions, caption_lengths)
                targets = captions[:, 1:]
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                test_batches += 1
                
                # 生成caption（用于评估）
                start_idx = self.vocabulary.get_start_idx()
                end_idx = self.vocabulary.get_end_idx()
                
                generated_captions = self.model.generate(
                    images, start_idx, end_idx, max_length=self.max_caption_length
                )
                
                # 处理每个样本
                for i, image_name in enumerate(image_names):
                    # 获取真实caption（从dataset中获取的所有参考caption）
                    if image_name in references_raw:
                        # 将原始caption字符串转换为词汇列表
                        ref_caption_lists = []
                        for raw_caption in references_raw[image_name]:
                            # 使用vocabulary的tokenize方法
                            tokens = self.vocabulary.tokenize(raw_caption)
                            ref_caption_lists.append(tokens)
                        references[image_name] = ref_caption_lists
                    else:
                        # 如果无法获取，使用batch中的caption
                        true_caption_indices = captions[i].cpu().numpy()
                        true_caption = self.vocabulary.indices_to_caption(
                            true_caption_indices.tolist(), remove_special=True
                        )
                        references[image_name] = [true_caption.split()]
                    
                    # 生成的caption
                    gen_caption_indices = generated_captions[i].cpu().numpy()
                    gen_caption = self.vocabulary.indices_to_caption(
                        gen_caption_indices.tolist(), remove_special=True
                    )
                    hypotheses[image_name] = gen_caption.split()
                    
                    # 保存前几个样本用于示例输出
                    if len(example_samples) < 10:
                        # 选择一个参考caption用于显示
                        if image_name in references_raw and len(references_raw[image_name]) > 0:
                            true_caption_display = references_raw[image_name][0]
                        else:
                            true_caption_indices = captions[i].cpu().numpy()
                            true_caption_display = self.vocabulary.indices_to_caption(
                                true_caption_indices.tolist(), remove_special=True
                            )
                        
                        example_samples.append({
                            'image_name': image_name,
                            'true_caption': true_caption_display,
                            'generated_caption': gen_caption
                        })
                
                test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_test_loss = test_loss / test_batches
        
        # 计算BLEU分数
        print("\n计算BLEU分数...")
        bleu_scores = calculate_bleu_scores(references, hypotheses)
        
        # 打印测试结果
        print("\n" + "=" * 60)
        print("测试结果")
        print("=" * 60)
        print(f"测试损失: {avg_test_loss:.4f}")
        print(f"\nBLEU分数:")
        for metric, score in bleu_scores.items():
            print(f"  {metric}: {score:.4f}")
        
        # 打印示例caption
        print(f"\n示例生成结果（前 {len(example_samples)} 个样本）:")
        print("-" * 60)
        for i, sample in enumerate(example_samples, 1):
            print(f"\n样本 {i}:")
            print(f"  图像: {sample['image_name']}")
            print(f"  真实描述: {sample['true_caption']}")
            print(f"  生成描述: {sample['generated_caption']}")
        
        print("\n" + "=" * 60)
        
        return {
            'test_loss': avg_test_loss,
            'bleu_scores': bleu_scores,
            'example_samples': example_samples
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise RuntimeError("模型未构建，无法保存")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': len(self.vocabulary) if self.vocabulary else None,
            'max_caption_length': self.max_caption_length
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer: Any, 
                       scheduler: Optional[Any] = None, best_val_loss: Optional[float] = None):
        """
        保存训练checkpoint（包含模型、优化器、scheduler等完整状态）
        
        Args:
            filepath: checkpoint保存路径
            epoch: 当前epoch
            optimizer: 优化器对象
            scheduler: 学习率调度器（可选）
            best_val_loss: 最佳验证损失（可选）
        """
        if self.model is None:
            raise RuntimeError("模型未构建，无法保存checkpoint")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_history': self.train_history,
            'vocab_size': len(self.vocabulary) if self.vocabulary else None,
            'max_caption_length': self.max_caption_length,
            'best_val_loss': best_val_loss,
            'training_params': self.training_params
        }
        
        # 如果有scheduler，保存scheduler状态
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str, vocab_size: int):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 从checkpoint恢复max_caption_length（如果存在）
        if 'max_caption_length' in checkpoint:
            self.max_caption_length = checkpoint['max_caption_length']
        
        # 构建模型（使用保存的参数）
        self.model = ImageCaptionModel(vocab_size=vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self._mark_model_ready()
        print(f"模型已从 {filepath} 加载")
        print(f"词汇表大小: {vocab_size}")
        print(f"Caption最大长度: {self.max_caption_length}")
    
    def load_checkpoint_model(self, filepath: str):
        """
        从checkpoint加载模型（仅加载模型，用于测试）
        与load_model不同，这个方法会从checkpoint中读取vocab_size等参数
        
        Args:
            filepath: checkpoint文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 从checkpoint获取vocab_size
        if 'vocab_size' in checkpoint:
            vocab_size = checkpoint['vocab_size']
        else:
            if self.vocabulary is None:
                raise RuntimeError("无法从checkpoint获取vocab_size，且vocabulary未设置")
            vocab_size = len(self.vocabulary)
        
        # 从checkpoint恢复max_caption_length（如果存在）
        if 'max_caption_length' in checkpoint:
            self.max_caption_length = checkpoint['max_caption_length']
        
        # 从checkpoint获取模型参数（如果checkpoint中有）
        # 否则需要手动指定参数
        embed_dim = 512
        hidden_dim = 512
        num_layers = 1
        dropout = 0.5
        freeze_encoder = False
        
        # 构建模型
        self.model = ImageCaptionModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_caption_length=self.max_caption_length,
            dropout=dropout,
            freeze_encoder=freeze_encoder
        )
        
        # 加载模型状态
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Checkpoint中未找到model_state_dict")
        
        self.model = self.model.to(self.device)
        self._mark_model_ready()
        
        # 如果有训练历史，也恢复（用于查看训练进度）
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        epoch = checkpoint.get('epoch', '未知')
        print(f"Checkpoint模型已从 {filepath} 加载")
        print(f"  Epoch: {epoch}")
        print(f"  词汇表大小: {vocab_size}")
        print(f"  Caption最大长度: {self.max_caption_length}")
        if 'best_val_loss' in checkpoint:
            print(f"  最佳验证损失: {checkpoint['best_val_loss']:.4f}")
    
    def load_checkpoint(self, filepath: str, optimizer: Any, scheduler: Optional[Any] = None) -> dict:
        """
        加载训练checkpoint并恢复训练状态
        
        Args:
            filepath: checkpoint文件路径
            optimizer: 优化器对象（需要先创建）
            scheduler: 学习率调度器对象（可选，需要先创建）
            
        Returns:
            包含恢复的训练状态的字典
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 恢复模型状态
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复scheduler状态
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练历史
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        # 恢复训练参数
        if 'training_params' in checkpoint:
            self.training_params = checkpoint['training_params']
        
        # 恢复max_caption_length
        if 'max_caption_length' in checkpoint:
            self.max_caption_length = checkpoint['max_caption_length']
        
        return checkpoint
