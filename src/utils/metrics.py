"""
评估指标计算模块
包括BLEU分数计算（使用pycocoevalcap或自实现版本）
"""
import numpy as np
from typing import List, Dict

try:
    from pycocoevalcap.bleu.bleu import Bleu
    PYCOCO_AVAILABLE = True
except ImportError:
    PYCOCO_AVAILABLE = False
    print("警告: pycocoevalcap未安装，将使用简化版BLEU计算")


def calculate_bleu_scores(references: Dict[str, List[List[str]]], 
                          hypotheses: Dict[str, List[str]]) -> Dict[str, float]:
    """
    计算BLEU分数（使用pycocoevalcap）
    
    Args:
        references: 参考caption字典，格式 {image_id: [list of reference captions]}
        hypotheses: 生成caption字典，格式 {image_id: [generated caption]}
        
    Returns:
        bleu_scores: BLEU-1, BLEU-2, BLEU-3, BLEU-4分数字典
    """
    if PYCOCO_AVAILABLE:
        return _calculate_bleu_pycoco(references, hypotheses)
    else:
        return _calculate_bleu_simple(references, hypotheses)


def _calculate_bleu_pycoco(references: Dict[str, List[List[str]]], 
                           hypotheses: Dict[str, List[str]]) -> Dict[str, float]:
    """使用pycocoevalcap计算BLEU"""
    # 转换格式：pycocoevalcap需要的格式
    # references: {image_id: [['word1', 'word2', ...], ['word1', 'word2', ...], ...]}
    # hypotheses: {image_id: ['word1', 'word2', ...]}  (注意：必须是单个列表，不是列表的列表)
    
    # 确保references格式正确（列表的列表）
    refs = {}
    for img_id, ref_list in references.items():
        if not ref_list:
            continue  # 跳过空的reference
        # 确保ref_list是列表的列表
        if isinstance(ref_list[0], str):
            # 如果第一个元素是字符串，说明ref_list是字符串列表，需要转换为列表的列表
            refs[img_id] = [ref.split() if isinstance(ref, str) else ref for ref in ref_list]
        else:
            # 已经是列表的列表
            refs[img_id] = ref_list
    
    # 确保hypotheses格式正确（单个词汇列表，不是列表的列表）
    hyps = {}
    for img_id, hyp in hypotheses.items():
        if isinstance(hyp, str):
            # 如果是字符串，转换为列表
            hyps[img_id] = hyp.split()
        elif isinstance(hyp, list):
            # 如果是列表，检查是否是列表的列表
            if hyp and isinstance(hyp[0], list):
                # 如果是列表的列表，取第一个（或合并）
                hyps[img_id] = hyp[0] if hyp else []
            else:
                # 已经是词汇列表
                hyps[img_id] = hyp
        else:
            # 其他类型，跳过
            continue
        
        # 确保hypothesis不为空
        if not hyps[img_id]:
            # 如果为空，添加一个占位符
            hyps[img_id] = ['<empty>']
    
    # 确保refs和hyps都有相同的keys
    common_keys = set(refs.keys()) & set(hyps.keys())
    if not common_keys:
        # 如果没有共同的keys，返回默认值
        return {
            'BLEU-1': 0.0,
            'BLEU-2': 0.0,
            'BLEU-3': 0.0,
            'BLEU-4': 0.0
        }
    
    # 只使用共同的keys
    refs = {k: refs[k] for k in common_keys}
    hyps = {k: hyps[k] for k in common_keys}
    
    # 计算BLEU
    try:
        scorer = Bleu(4)  # 计算BLEU-1到BLEU-4
        scores, _ = scorer.compute_score(refs, hyps)
        
        return {
            'BLEU-1': scores[0] if isinstance(scores, (list, tuple)) and len(scores) > 0 else 0.0,
            'BLEU-2': scores[1] if isinstance(scores, (list, tuple)) and len(scores) > 1 else 0.0,
            'BLEU-3': scores[2] if isinstance(scores, (list, tuple)) and len(scores) > 2 else 0.0,
            'BLEU-4': scores[3] if isinstance(scores, (list, tuple)) and len(scores) > 3 else 0.0
        }
    except Exception as e:
        print(f"BLEU计算错误: {e}")
        print(f"References keys数量: {len(refs)}")
        print(f"Hypotheses keys数量: {len(hyps)}")
        if refs:
            print(f"示例reference: {list(refs.values())[0]}")
        if hyps:
            print(f"示例hypothesis: {list(hyps.values())[0]}")
        # 如果出错，使用简化版BLEU
        return _calculate_bleu_simple(references, hypotheses)


def _calculate_bleu_simple(references: Dict[str, List[List[str]]], 
                          hypotheses: Dict[str, List[str]]) -> Dict[str, float]:
    """简化版BLEU计算（不依赖外部库）"""
    def ngram_count(tokens, n):
        """计算n-gram计数"""
        count = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            count[ngram] = count.get(ngram, 0) + 1
        return count
    
    def precision_n(references, hypothesis, n):
        """计算n-gram精确度"""
        hyp_ngrams = ngram_count(hypothesis, n)
        if len(hyp_ngrams) == 0:
            return 0.0
        
        matches = 0
        for ngram in hyp_ngrams:
            max_ref_count = max([ngram_count(ref, n).get(ngram, 0) for ref in references])
            matches += min(hyp_ngrams[ngram], max_ref_count)
        
        return matches / len(hyp_ngrams)
    
    def brevity_penalty(references, hypothesis):
        """计算简洁性惩罚"""
        ref_lengths = [len(ref) for ref in references]
        hyp_length = len(hypothesis)
        closest_ref_length = min(ref_lengths, key=lambda x: abs(x - hyp_length))
        
        if hyp_length > closest_ref_length:
            return 1.0
        return np.exp(1 - closest_ref_length / hyp_length)
    
    # 计算每个样本的BLEU分数
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    
    for img_id in hypotheses:
        if img_id not in references:
            continue
        
        hyp = hypotheses[img_id]
        if isinstance(hyp, str):
            hyp = hyp.split()
        
        refs = references[img_id]
        refs = [ref.split() if isinstance(ref, str) else ref for ref in refs]
        
        bp = brevity_penalty(refs, hyp)
        
        for n in range(1, 5):
            precisions = [precision_n(refs, hyp, i) for i in range(1, n + 1)]
            if all(p > 0 for p in precisions):
                bleu_n = bp * (np.prod(precisions) ** (1.0 / n))
            else:
                bleu_n = 0.0
            bleu_scores[n].append(bleu_n)
    
    # 计算平均BLEU分数
    return {
        'BLEU-1': np.mean(bleu_scores[1]) if bleu_scores[1] else 0.0,
        'BLEU-2': np.mean(bleu_scores[2]) if bleu_scores[2] else 0.0,
        'BLEU-3': np.mean(bleu_scores[3]) if bleu_scores[3] else 0.0,
        'BLEU-4': np.mean(bleu_scores[4]) if bleu_scores[4] else 0.0
    }


