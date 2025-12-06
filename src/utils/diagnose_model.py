"""
模型诊断工具
用于检查模型是否陷入局部最优点
"""
import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np


def check_model_output_distribution(model, test_loader, vocabulary, device, num_samples=100):
    """
    检查模型输出的词汇分布，看是否总是预测相同的词
    
    Args:
        model: 图像描述模型
        test_loader: 测试数据加载器
        vocabulary: 词汇表
        device: 计算设备
        
    Returns:
        dict: 包含词汇分布统计的字典
    """
    model.eval()
    
    # 统计每个位置预测的词汇分布
    position_vocab_counts = {}  # {position: {vocab_idx: count}}
    all_predictions = []
    
    start_idx = vocabulary.get_start_idx()
    end_idx = vocabulary.get_end_idx()
    
    with torch.no_grad():
        sample_count = 0
        for images, captions, caption_lengths, image_names in test_loader:
            if sample_count >= num_samples:
                break
                
            images = images.to(device)
            
            # 生成caption
            generated = model.generate(images, start_idx, end_idx, max_length=25)
            
            # 统计每个位置的词汇分布
            for i in range(generated.size(0)):
                if sample_count >= num_samples:
                    break
                    
                gen_seq = generated[i].cpu().numpy()
                all_predictions.append(gen_seq)
                
                for pos, vocab_idx in enumerate(gen_seq):
                    if pos not in position_vocab_counts:
                        position_vocab_counts[pos] = {}
                    if vocab_idx not in position_vocab_counts[pos]:
                        position_vocab_counts[pos][vocab_idx] = 0
                    position_vocab_counts[pos][vocab_idx] += 1
                
                sample_count += 1
    
    # 分析结果
    results = {
        'position_diversity': {},  # 每个位置的词汇多样性（不同词汇数）
        'most_common_words': {},  # 每个位置最常见的词
        'repetition_rate': 0.0,  # 重复率
    }
    
    # 计算每个位置的多样性
    for pos in sorted(position_vocab_counts.keys()):
        vocab_counts = position_vocab_counts[pos]
        total = sum(vocab_counts.values())
        diversity = len(vocab_counts)  # 不同词汇数
        
        # 找到最常见的词
        most_common_idx = max(vocab_counts.items(), key=lambda x: x[1])[0]
        most_common_word = vocabulary.idx2word.get(most_common_idx, f"UNK_{most_common_idx}")
        most_common_freq = vocab_counts[most_common_idx] / total
        
        results['position_diversity'][pos] = {
            'unique_words': diversity,
            'most_common_word': most_common_word,
            'most_common_freq': most_common_freq,
            'total_samples': total
        }
    
    # 计算重复率（所有样本生成相同序列的比例）
    if len(all_predictions) > 0:
        unique_sequences = len(set(tuple(seq) for seq in all_predictions))
        repetition_rate = 1.0 - (unique_sequences / len(all_predictions))
        results['repetition_rate'] = repetition_rate
        results['unique_sequences'] = unique_sequences
        results['total_samples'] = len(all_predictions)
    
    return results


def check_gradient_flow(model):
    """
    检查梯度流，看是否有梯度消失或爆炸问题
    
    Returns:
        dict: 包含梯度统计的字典
    """
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            
            grad_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': param.grad.max().item(),
                'min': param.grad.min().item()
            }
    
    return grad_stats


def check_image_features(model, test_loader, device, num_samples=10):
    """
    检查图像特征是否正确提取，以及不同图像的特征是否不同
    
    Returns:
        dict: 包含图像特征统计的字典
    """
    model.eval()
    
    image_features_list = []
    
    with torch.no_grad():
        sample_count = 0
        for images, captions, caption_lengths, image_names in test_loader:
            if sample_count >= num_samples:
                break
                
            images = images.to(device)
            features = model.encode_image(images)
            image_features_list.append(features.cpu())
            sample_count += images.size(0)
    
    if len(image_features_list) == 0:
        return {}
    
    all_features = torch.cat(image_features_list, dim=0)
    
    # 计算特征统计
    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0)
    
    # 计算不同图像特征之间的相似度（余弦相似度）
    if all_features.size(0) > 1:
        # 归一化特征
        normalized_features = nn.functional.normalize(all_features, p=2, dim=1)
        # 计算相似度矩阵
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        # 计算平均相似度（排除对角线）
        mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool)
        avg_similarity = similarity_matrix[mask].mean().item()
    else:
        avg_similarity = 1.0
    
    return {
        'feature_mean': feature_mean.numpy(),
        'feature_std': feature_std.numpy(),
        'avg_similarity': avg_similarity,
        'num_samples': all_features.size(0),
        'feature_dim': all_features.size(1)
    }


def print_diagnosis(results: Dict):
    """
    打印诊断结果
    """
    print("\n" + "=" * 60)
    print("模型诊断结果")
    print("=" * 60)
    
    if 'output_distribution' in results:
        print("\n[1] 输出分布分析:")
        dist = results['output_distribution']
        
        print(f"  重复率: {dist['repetition_rate']:.2%}")
        print(f"  唯一序列数: {dist.get('unique_sequences', 0)} / {dist.get('total_samples', 0)}")
        
        print("\n  各位置词汇分布:")
        for pos in sorted(dist['position_diversity'].keys())[:10]:  # 只显示前10个位置
            info = dist['position_diversity'][pos]
            print(f"    位置 {pos}:")
            print(f"      不同词汇数: {info['unique_words']}")
            print(f"      最常见词: '{info['most_common_word']}' (频率: {info['most_common_freq']:.2%})")
    
    if 'gradient_flow' in results:
        print("\n[2] 梯度流分析:")
        grad_stats = results['gradient_flow']
        
        if len(grad_stats) == 0:
            print("  警告: 未检测到梯度（模型可能未训练或处于eval模式）")
        else:
            print(f"  检测到 {len(grad_stats)} 个参数组")
            
            # 统计梯度范数
            grad_norms = [stat['norm'] for stat in grad_stats.values()]
            print(f"  平均梯度范数: {np.mean(grad_norms):.6f}")
            print(f"  最大梯度范数: {np.max(grad_norms):.6f}")
            print(f"  最小梯度范数: {np.min(grad_norms):.6f}")
            
            # 检查梯度消失/爆炸
            if np.max(grad_norms) > 100:
                print("  ⚠️  警告: 检测到梯度爆炸（梯度范数 > 100）")
            if np.min(grad_norms) < 1e-6:
                print("  ⚠️  警告: 检测到梯度消失（梯度范数 < 1e-6）")
    
    if 'image_features' in results:
        print("\n[3] 图像特征分析:")
        feat_stats = results['image_features']
        
        if feat_stats:
            print(f"  样本数: {feat_stats['num_samples']}")
            print(f"  特征维度: {feat_stats['feature_dim']}")
            print(f"  平均特征相似度: {feat_stats['avg_similarity']:.4f}")
            
            if feat_stats['avg_similarity'] > 0.95:
                print("  ⚠️  警告: 图像特征过于相似（>0.95），可能所有图像被编码为相同特征")
    
    print("\n" + "=" * 60)
    
    # 给出建议
    print("\n建议:")
    suggestions = []
    
    if 'output_distribution' in results:
        dist = results['output_distribution']
        if dist.get('repetition_rate', 0) > 0.5:
            suggestions.append("• 模型陷入局部最优点，建议：")
            suggestions.append("  - 降低学习率（当前0.003，可尝试0.001或0.0005）")
            suggestions.append("  - 检查图像特征是否正确提取")
            suggestions.append("  - 尝试不同的初始化方法")
            suggestions.append("  - 增加dropout或使用更强的正则化")
    
    if 'gradient_flow' in results:
        grad_stats = results['gradient_flow']
        if grad_stats:
            grad_norms = [stat['norm'] for stat in grad_stats.values()]
            if np.max(grad_norms) > 100:
                suggestions.append("• 检测到梯度爆炸，建议降低学习率或使用梯度裁剪")
            if np.min(grad_norms) < 1e-6:
                suggestions.append("• 检测到梯度消失，建议检查模型架构或使用残差连接")
    
    if 'image_features' in results:
        feat_stats = results['image_features']
        if feat_stats and feat_stats.get('avg_similarity', 0) > 0.95:
            suggestions.append("• 图像特征过于相似，建议：")
            suggestions.append("  - 检查CNN编码器是否正确加载预训练权重")
            suggestions.append("  - 检查图像预处理是否正确")
            suggestions.append("  - 考虑解冻CNN编码器进行微调")
    
    if not suggestions:
        suggestions.append("• 模型状态看起来正常，继续训练观察效果")
    
    for suggestion in suggestions:
        print(suggestion)
    
    print("=" * 60 + "\n")




