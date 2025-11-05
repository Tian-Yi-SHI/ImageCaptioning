"""
图像描述生成模型：CNN-LSTM架构
编码器：ResNet18（预训练）
解码器：LSTM
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class ImageCaptionModel(nn.Module):
    """
    图像描述生成模型
    
    架构：
    - CNN编码器（ResNet18）：提取图像特征
    - LSTM解码器：生成文本序列
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 1,
                 max_caption_length: int = 30,
                 dropout: float = 0.5,
                 freeze_encoder: bool = False):
        """
        初始化模型
        
        Args:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            max_caption_length: 最大caption长度
            dropout: Dropout概率
        """
        super(ImageCaptionModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_caption_length = max_caption_length
        self.dropout = dropout
        self.freeze_encoder = freeze_encoder  # 论文建议冻结CNN
        
        # CNN编码器：使用预训练的ResNet18
        # 使用weights参数替代已弃用的pretrained参数
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 移除最后的全连接层和平均池化层
        modules = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*modules)
        
        # 获取ResNet18的特征图维度
        # ResNet18最后卷积层的输出是 (batch, 512, H, W)
        # 这里我们使用全局平均池化得到 (batch, 512) 的特征向量
        self.encoder_pool = nn.AdaptiveAvgPool2d((1, 1))
        encoder_output_dim = 512
        
        # 将图像特征投影到LSTM输入维度
        self.image_projection = nn.Linear(encoder_output_dim, hidden_dim)
        
        # 如果冻结CNN，也冻结image_projection和encoder
        if freeze_encoder:
            # 冻结CNN编码器参数
            for param in self.encoder.parameters():
                param.requires_grad = False
            # 冻结图像投影层
            for param in self.image_projection.parameters():
                param.requires_grad = False
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM解码器
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层：将LSTM输出映射到词汇表
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化参数
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型参数"""
        # 嵌入层和输出层使用xavier初始化
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        
        # 图像投影层初始化
        nn.init.xavier_uniform_(self.image_projection.weight)
        nn.init.constant_(self.image_projection.bias, 0.0)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像特征
        
        Args:
            images: (batch_size, 3, H, W) 图像tensor
            freeze_encoder: 是否冻结编码器（论文建议冻结CNN）
            
        Returns:
            image_features: (batch_size, hidden_dim) 图像特征向量
        """
        # 通过ResNet编码器
        # 论文：冻结CNN权重会带来更好的效果
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(images)  # (batch_size, 512, H', W')
        else:
            features = self.encoder(images)  # (batch_size, 512, H', W')
        
        # 全局平均池化
        features = self.encoder_pool(features)  # (batch_size, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 512)
        
        # 投影到LSTM维度
        # 如果冻结了CNN，image_projection也会被冻结，但前向传播仍需要计算梯度（用于LSTM）
        image_features = self.image_projection(features)  # (batch_size, hidden_dim)
        
        return image_features
    
    def decode_step(self, 
                    word_embeddings: torch.Tensor,
                    hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        LSTM解码一步
        
        Args:
            word_embeddings: (batch_size, 1, embed_dim) 当前词嵌入
            hidden_state: LSTM的隐藏状态 (h, c)
            
        Returns:
            output: (batch_size, 1, hidden_dim) LSTM输出
            hidden_state: 更新后的隐藏状态
        """
        # LSTM前向传播
        lstm_out, hidden_state = self.lstm(word_embeddings, hidden_state)
        
        return lstm_out, hidden_state
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor, 
                caption_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播（训练模式）
        
        Args:
            images: (batch_size, 3, H, W) 图像tensor
            captions: (batch_size, max_length) caption索引tensor（包含START和END）
            caption_lengths: (batch_size,) 每个caption的实际长度
            
        Returns:
            outputs: (batch_size, max_length-1, vocab_size) 每个时间步的词汇概率分布
        """
        batch_size = images.size(0)
        
        # 编码图像
        image_features = self.encode_image(images)  # (batch_size, hidden_dim)
        
        # 初始化LSTM隐藏状态（使用图像特征）
        h0 = image_features.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch_size, hidden_dim)
        c0 = torch.zeros_like(h0)
        hidden_state = (h0, c0)
        
        # 词嵌入（去掉最后一个token，因为我们预测的是下一个词）
        # captions: (batch_size, max_length)
        # 输入序列：captions[:, :-1]，去掉最后一个token
        # 目标序列：captions[:, 1:]，去掉第一个token（START）
        input_captions = captions[:, :-1]  # (batch_size, max_length-1)
        word_embeddings = self.embedding(input_captions)  # (batch_size, max_length-1, embed_dim)
        word_embeddings = self.dropout_layer(word_embeddings)
        
        # LSTM解码
        lstm_out, _ = self.lstm(word_embeddings, hidden_state)  # (batch_size, max_length-1, hidden_dim)
        lstm_out = self.dropout_layer(lstm_out)
        
        # 输出层
        outputs = self.fc(lstm_out)  # (batch_size, max_length-1, vocab_size)
        
        return outputs
    
    def generate(self, images: torch.Tensor, 
                 start_idx: int,
                 end_idx: int,
                 max_length: Optional[int] = None,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        生成caption（推理模式）
        
        Args:
            images: (batch_size, 3, H, W) 图像tensor
            start_idx: START token的索引
            end_idx: END token的索引
            max_length: 最大生成长度（如果为None，使用self.max_caption_length）
            temperature: 温度参数，控制生成的随机性（1.0为标准softmax）
            
        Returns:
            generated_captions: (batch_size, actual_length) 生成的caption索引序列
        """
        if max_length is None:
            max_length = self.max_caption_length
        
        batch_size = images.size(0)
        device = images.device
        
        # 编码图像
        image_features = self.encode_image(images)  # (batch_size, hidden_dim)
        
        # 初始化LSTM隐藏状态
        h0 = image_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        hidden_state = (h0, c0)
        
        # 初始化生成序列（从START token开始）
        generated_captions = []
        current_word = torch.full((batch_size,), start_idx, dtype=torch.long, device=device)
        
        for step in range(max_length):
            # 词嵌入
            word_embedding = self.embedding(current_word).unsqueeze(1)  # (batch_size, 1, embed_dim)
            
            # LSTM解码一步
            lstm_out, hidden_state = self.decode_step(word_embedding, hidden_state)
            
            # 输出层
            output = self.fc(lstm_out.squeeze(1))  # (batch_size, vocab_size)
            
            # 应用温度
            if temperature != 1.0:
                output = output / temperature
            
            # 采样下一个词（使用greedy decoding）
            current_word = torch.argmax(output, dim=1)  # (batch_size,)
            generated_captions.append(current_word.clone())  # 使用clone避免引用问题
            
            # 检查是否所有序列都结束了（遇到END token）
            # 注意：这里检查的是所有batch都结束，但实际上应该分别处理每个序列
            # 改进：使用mask来跟踪每个序列是否结束
            if step == 0:
                # 创建mask来跟踪哪些序列还在生成
                continue_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            # 更新mask：如果遇到END token，标记为已完成
            continue_mask = continue_mask & (current_word != end_idx)
            
            # 如果所有序列都结束了，提前退出
            if not continue_mask.any():
                break
        
        # 拼接生成序列
        generated_captions = torch.stack(generated_captions, dim=1)  # (batch_size, actual_length)
        
        return generated_captions


