"""
Transformer架构的图像描述生成模型 - 使用Nyström位置编码

编码器：CNN + Transformer Encoder（带Nyström位置编码）
解码器：Transformer Decoder

位置编码创新：
- 使用Nyström方法近似点之间的两两距离
- 选择锚点，用它们来计算RBF距离
- 通过低秩近似得到位置编码
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple


class NystromPositionalEncoding(nn.Module):
    """
    Nyström方法的位置编码模块
    
    使用Nyström近似来计算图像特征的两两距离，并用其作为位置编码
    """
    
    def __init__(self, feature_dim: int = 256, num_anchors: int = 16, 
                 sigma: float = 1.0, pos_dim: int = 256):
        """
        初始化Nyström位置编码模块
        
        Args:
            feature_dim: 特征维度
            num_anchors: 锚点数量
            sigma: RBF高斯核的带宽参数
            pos_dim: 位置编码最终输出维度
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_anchors = num_anchors
        self.sigma = sigma
        self.pos_dim = pos_dim
        
        # 位置编码投影层
        self.pos_projection = nn.Linear(num_anchors, pos_dim)
        
    def _compute_rbf_kernel(self, positions: torch.Tensor, anchors: torch.Tensor, 
                           sigma: float) -> torch.Tensor:
        """
        计算RBF核矩阵
        
        Args:
            positions: (n, 2) 空间位置坐标
            anchors: (m, 2) 锚点坐标
            sigma: RBF高斯核的带宽参数
            
        Returns:
            K: (n, m) RBF核矩阵
        """
        # 计算欧氏距离
        # positions: (n, 2), anchors: (m, 2)
        # diff: (n, m, 2)
        diff = positions.unsqueeze(1) - anchors.unsqueeze(0)  # (n, m, 2)
        
        # 计算距离的平方
        dist_sq = torch.sum(diff ** 2, dim=2)  # (n, m)
        
        # RBF核: exp(-dist^2 / (2*sigma^2))
        K = torch.exp(-dist_sq / (2 * sigma ** 2))  # (n, m)
        
        return K
    
    def _select_anchors(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        选择锚点（均匀采样空间位置）
        
        Args:
            H: 特征图高度
            W: 特征图宽度
            device: 设备
            
        Returns:
            anchors: (num_anchors, 2) 锚点坐标 (归一化到[-1, 1])
        """
        # 根据锚点数量确定采样间隔
        num_anchors_h = max(1, int(math.sqrt(self.num_anchors * H / W)))
        num_anchors_w = max(1, self.num_anchors // num_anchors_h)
        
        # 生成均匀采样的锚点
        anchor_h = torch.linspace(-1, 1, num_anchors_h, device=device)
        anchor_w = torch.linspace(-1, 1, num_anchors_w, device=device)
        
        anchor_grid_h, anchor_grid_w = torch.meshgrid(anchor_h, anchor_w, indexing='ij')
        anchors = torch.stack([anchor_grid_w.flatten(), anchor_grid_h.flatten()], dim=1)  # (*, 2)
        
        # 调整到目标锚点数量
        if anchors.shape[0] > self.num_anchors:
            idx = torch.randperm(anchors.shape[0], device=device)[:self.num_anchors]
            anchors = anchors[idx]
        elif anchors.shape[0] < self.num_anchors:
            # 如果不足，随机复制一些锚点
            shortage = self.num_anchors - anchors.shape[0]
            idx = torch.randint(0, anchors.shape[0], (shortage,), device=device)
            additional_anchors = anchors[idx]
            # 添加少量噪声以区分
            additional_anchors = additional_anchors + torch.randn_like(additional_anchors) * 0.01
            anchors = torch.cat([anchors, additional_anchors], dim=0)
        
        return anchors[:self.num_anchors]
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: (B, C, H, W) 特征图
            
        Returns:
            nystrom_pos_encoding: (B, H*W, pos_dim) Nyström位置编码
        """
        B, C, H, W = features.shape
        device = features.device
        
        # 1. 生成位置坐标网格
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 拼接为位置坐标 (H*W, 2)
        positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        
        # 2. 选择锚点
        anchors = self._select_anchors(H, W, device)  # (num_anchors, 2)
        
        # 3. 计算所有点到锚点的RBF距离
        K = self._compute_rbf_kernel(positions, anchors, self.sigma)  # (H*W, num_anchors)
        
        # 4. 投影到位置编码维度
        pos_encoding = self.pos_projection(K)  # (H*W, pos_dim)
        
        # 5. 扩展到batch维度
        pos_encoding = pos_encoding.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, pos_dim)
        
        return pos_encoding


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层（用于图像特征）"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: (B, seq_len, d_model)
            src_mask: 注意力mask（可选）
        """
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层（用于文本生成）"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        # Masked self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Cross-attention (图像特征 -> 文本)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: (B, tgt_len, d_model) 目标序列（文本）
            memory: (B, src_len, d_model) 编码后的图像特征
            tgt_mask: 目标序列mask（用于防止看到未来信息）
            memory_mask: 图像特征mask（可选）
        """
        # Masked self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention (图像 -> 文本)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class NystromTransformerModel(nn.Module):
    """
    Transformer架构的图像描述生成模型 - 使用Nyström位置编码
    
    架构：
    - CNN编码器：提取图像特征图
    - Nyström位置编码：使用锚点和RBF距离
    - Transformer编码器：处理图像特征
    - Transformer解码器：生成文本序列
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_caption_length: int = 30,
                 pos_dim: int = 256,
                 feature_dim: int = 256,
                 freeze_encoder: bool = False,
                 threshold_mode: str = 'adaptive',
                 num_anchors: int = 16,
                 sigma: float = 1.0):
        """
        初始化模型
        
        Args:
            vocab_size: 词汇表大小
            d_model: Transformer维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: FFN维度
            dropout: Dropout概率
            max_caption_length: 最大caption长度
            pos_dim: 位置编码维度
            feature_dim: 特征维度
            freeze_encoder: 是否冻结CNN编码器
            threshold_mode: 位置编码阈值模式（兼容参数）
            num_anchors: Nyström方法的锚点数量
            sigma: RBF核的带宽参数
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_caption_length = max_caption_length
        self.freeze_encoder = freeze_encoder
        
        # CNN编码器：使用ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 移除最后的全连接层和平均池化层
        modules = list(resnet.children())[:-2]
        self.cnn_encoder = nn.Sequential(*modules)
        
        # ResNet18的输出是 (B, 512, H', W')
        cnn_output_dim = 512
        
        # 特征投影层：将CNN特征投影到d_model
        self.feature_projection = nn.Linear(cnn_output_dim, d_model)
        
        # Nyström位置编码模块
        self.pos_encoding = NystromPositionalEncoding(
            feature_dim=feature_dim,
            num_anchors=num_anchors,
            sigma=sigma,
            pos_dim=pos_dim
        )
        
        # 位置编码投影到d_model
        self.pos_projection = nn.Linear(pos_dim, d_model)
        
        # Transformer编码器
        encoder_layers = [
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 文本位置编码（标准Transformer PE）
        self.text_pos_encoding = nn.Parameter(
            torch.randn(max_caption_length, d_model)
        )
        
        # Transformer解码器
        decoder_layers = [
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ]
        self.transformer_decoder = nn.ModuleList(decoder_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 如果冻结CNN，冻结相关参数
        if freeze_encoder:
            for param in self.cnn_encoder.parameters():
                param.requires_grad = False
            for param in self.feature_projection.parameters():
                param.requires_grad = False
        
        # 初始化参数
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型参数"""
        # 嵌入层
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # 文本位置编码
        nn.init.normal_(self.text_pos_encoding, mean=0, std=0.02)
        
        # 输出层
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0.0)
        
        # 特征投影层
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.constant_(self.feature_projection.bias, 0.0)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像特征
        
        Args:
            images: (B, 3, H, W) 图像tensor
            
        Returns:
            encoded_features: (B, seq_len, d_model) 编码后的图像特征
        """
        B = images.size(0)
        
        # CNN编码
        if self.freeze_encoder:
            with torch.no_grad():
                cnn_features = self.cnn_encoder(images)  # (B, 512, H', W')
        else:
            cnn_features = self.cnn_encoder(images)  # (B, 512, H', W')
        
        B, C, H, W = cnn_features.shape
        
        # 计算Nyström位置编码
        pos_encoding = self.pos_encoding(cnn_features)  # (B, H*W, pos_dim)
        pos_encoding = self.pos_projection(pos_encoding)  # (B, H*W, d_model)
        
        # 投影CNN特征到d_model
        features = cnn_features.permute(0, 2, 3, 1)  # (B, H, W, C)
        features = features.reshape(B, H * W, C)  # (B, H*W, C)
        features = self.feature_projection(features)  # (B, H*W, d_model)
        
        # 融合特征和位置编码
        features = features + pos_encoding  # (B, H*W, d_model)
        
        # Transformer编码器
        for layer in self.transformer_encoder:
            features = layer(features)
        
        return features
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成因果mask（防止看到未来信息）"""
        mask = torch.triu(torch.ones(sz, sz, device=next(self.parameters()).device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor,
                caption_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播（训练模式）
        
        Args:
            images: (B, 3, H, W) 图像tensor
            captions: (B, max_length) caption索引tensor（包含START和END）
            caption_lengths: (B,) 每个caption的实际长度（未使用，为兼容性保留）
            
        Returns:
            outputs: (B, max_length-1, vocab_size) 每个时间步的词汇概率分布
        """
        B = images.size(0)
        device = images.device
        
        # 编码图像
        image_features = self.encode_image(images)  # (B, H*W, d_model)
        
        # 词嵌入（去掉最后一个token）
        input_captions = captions[:, :-1]  # (B, max_length-1)
        seq_len = input_captions.size(1)
        
        # 文本嵌入 + 位置编码
        text_emb = self.embedding(input_captions)  # (B, seq_len, d_model)
        text_pos = self.text_pos_encoding[:seq_len].unsqueeze(0)  # (1, seq_len, d_model)
        text_emb = text_emb + text_pos
        text_emb = self.dropout(text_emb)
        
        # 生成因果mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        
        # Transformer解码器
        decoder_output = text_emb
        for layer in self.transformer_decoder:
            decoder_output = layer(decoder_output, image_features, tgt_mask=tgt_mask)
        
        # 输出层
        outputs = self.output_projection(decoder_output)  # (B, seq_len, vocab_size)
        
        return outputs
    
    def generate(self, images: torch.Tensor,
                 start_idx: int,
                 end_idx: int,
                 max_length: Optional[int] = None,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        生成caption（推理模式）
        
        Args:
            images: (B, 3, H, W) 图像tensor
            start_idx: START token的索引
            end_idx: END token的索引
            max_length: 最大生成长度（如果为None，使用self.max_caption_length）
            temperature: 温度参数，控制生成的随机性
            
        Returns:
            generated_captions: (B, actual_length) 生成的caption索引序列
        """
        if max_length is None:
            max_length = self.max_caption_length
        
        B = images.size(0)
        device = images.device
        
        # 编码图像
        image_features = self.encode_image(images)  # (B, H*W, d_model)
        
        # 初始化生成序列
        generated_captions = []
        current_word = torch.full((B,), start_idx, dtype=torch.long, device=device)
        continue_mask = torch.ones(B, dtype=torch.bool, device=device)
        
        # 存储已生成的序列（用于mask）
        generated_sequence = [current_word.clone()]
        
        for step in range(max_length):
            # 当前序列长度
            seq_len = len(generated_sequence)
            
            # 构建输入序列
            input_seq = torch.stack(generated_sequence, dim=1)  # (B, seq_len)
            
            # 文本嵌入 + 位置编码
            text_emb = self.embedding(input_seq)  # (B, seq_len, d_model)
            text_pos = self.text_pos_encoding[:seq_len].unsqueeze(0)  # (1, seq_len, d_model)
            text_emb = text_emb + text_pos
            
            # 生成因果mask
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)
            
            # Transformer解码器
            decoder_output = text_emb
            for layer in self.transformer_decoder:
                decoder_output = layer(decoder_output, image_features, tgt_mask=tgt_mask)
            
            # 只使用最后一个时间步的输出
            last_output = decoder_output[:, -1, :]  # (B, d_model)
            
            # 输出层
            output = self.output_projection(last_output)  # (B, vocab_size)
            
            # 应用温度
            if temperature != 1.0:
                output = output / temperature
            
            # 采样下一个词（greedy decoding）
            next_word = torch.argmax(output, dim=1)  # (B,)
            
            # 只保存还在生成的序列的词
            next_word = torch.where(continue_mask, next_word,
                                   torch.full_like(next_word, end_idx))
            
            generated_captions.append(next_word.clone())
            generated_sequence.append(next_word.clone())
            
            # 更新mask
            continue_mask = continue_mask & (next_word != end_idx)
            
            # 如果所有序列都结束了，提前退出
            if not continue_mask.any():
                break
        
        # 拼接生成序列（去掉第一个START token）
        if len(generated_captions) > 0:
            generated_captions = torch.stack(generated_captions, dim=1)  # (B, actual_length)
        else:
            generated_captions = torch.full((B, 1), end_idx, dtype=torch.long, device=device)
        
        return generated_captions
