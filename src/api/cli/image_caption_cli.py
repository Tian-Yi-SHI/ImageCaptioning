"""
图像描述生成命令行工具
用于训练和测试模型
"""
import os
import sys
from pathlib import Path

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from config.config import *
from core.PipelineIC import PipelineIC
from core.data_processing import split_dataset, get_dataloaders
from utils.vocabulary import Vocabulary
from utils.collate_fn import collate_fn
from infrastructure.data.Flickr8KDataset import Flickr8KDataset
from infrastructure.data.TransformDatasetWrapper import TransformDatasetWrapper
from torchvision import transforms
from torch.utils.data import DataLoader


def build_vocabulary_from_dataset(dataset):
    """
    从数据集中构建词汇表
    
    Args:
        dataset: 数据集对象
        
    Returns:
        vocabulary: 构建好的词汇表
    """
    print("\n开始构建词汇表...")
    captions_list = []
    
    # 收集所有caption
    for i in range(len(dataset)):
        _, captions, _ = dataset[i]
        captions_list.append(captions)
    
    # 构建词汇表（按照《Show and Tell》论文方法）
    # 论文：仅保留出现至少5次的单词，并根据训练集实际情况自动调整词汇表大小
    # 设置一个较大的初始值，build_vocabulary会自动调整到实际需要的值
    vocabulary = Vocabulary(max_vocab_size=10000)  # 初始值，会自动调整
    vocabulary.build_vocabulary(
        captions_list, 
        min_word_freq=5,  # 论文：至少5次
        auto_adjust=True  # 自动根据训练集调整词汇表大小（论文方法）
    )
    
    return vocabulary


# 全局变量用于存储vocabulary（用于多进程）
_collate_fn_vocab = None
_collate_fn_max_length = 30

def _collate_fn_wrapper(batch):
    """模块级别的collate_fn包装器（用于多进程）"""
    global _collate_fn_vocab, _collate_fn_max_length
    if _collate_fn_vocab is None:
        raise RuntimeError("Vocabulary not set for collate_fn. Call set_collate_fn_vocab first.")
    return collate_fn(batch, _collate_fn_vocab, _collate_fn_max_length)

def set_collate_fn_vocab(vocabulary, max_caption_length=30):
    """
    设置collate_fn使用的vocabulary（用于多进程）
    
    Args:
        vocabulary: 词汇表对象
        max_caption_length: 最大caption长度
    """
    global _collate_fn_vocab, _collate_fn_max_length
    _collate_fn_vocab = vocabulary
    _collate_fn_max_length = max_caption_length


def main():
    """
    主函数：完整的训练和测试流程
    """
    print("=" * 60)
    print("图像描述生成系统 - 训练和测试")
    print("=" * 60)
    
    # ========== 1. 配置加载 ==========
    print("\n[1/7] 加载配置...")
    device = define_dev()
    folder_args, hyper_args, flag_args = read_config()
    
    print(f"  图像目录: {folder_args['folder_image']}")
    print(f"  Caption文件: {folder_args['file_caption']}")
    print(f"  模型保存目录: {folder_args['folder_model']}")
    print(f"  学习率: {hyper_args['lr']}")
    print(f"  批次大小: {hyper_args['batch_size']}")
    print(f"  训练轮数: {hyper_args['epoches']}")
    
    # 创建必要的目录
    Path(folder_args['folder_model']).mkdir(parents=True, exist_ok=True)
    Path(folder_args['folder_train_log']).mkdir(parents=True, exist_ok=True)
    Path(folder_args['folder_test_log']).mkdir(parents=True, exist_ok=True)
    
    # ========== 2. 数据加载 ==========
    print("\n[2/7] 加载数据集...")
    image_dir = folder_args['folder_image']
    caption_path = folder_args['file_caption']
    
    # 检查路径
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Caption文件不存在: {caption_path}")
    
    # 加载原始数据集
    raw_dataset = Flickr8KDataset(
        folder_path_image=image_dir,
        file_path_caption=caption_path
    )
    
    print(f"  数据集大小: {len(raw_dataset)} 张图像")
    
    # ========== 3. 构建词汇表 ==========
    print("\n[3/7] 构建词汇表...")
    vocabulary = build_vocabulary_from_dataset(raw_dataset)
    print(f"  词汇表大小: {len(vocabulary)}")
    
    # ========== 4. 数据预处理和分割 ==========
    print("\n[4/7] 数据预处理和分割...")
    
    # 定义图像变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 先应用变换到原始数据集，然后分割
    # 训练集使用训练变换
    train_dataset_wrapped = TransformDatasetWrapper(raw_dataset, train_transform)
    # 验证和测试集使用相同的变换
    val_test_dataset_wrapped = TransformDatasetWrapper(raw_dataset, val_test_transform)
    
    # 分割数据集（使用包装后的数据集，但需要保持原始数据集的引用）
    # 注意：split_dataset需要原始数据集，所以我们先分割原始数据集
    train_subset_raw, val_subset_raw, test_subset_raw = split_dataset(
        dataset=raw_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # 现在为每个subset应用transform
    # 由于Subset包装了原始数据集，我们需要创建一个新的wrapper
    # 但TransformDatasetWrapper不能直接包装Subset，所以我们需要手动处理
    # 解决方案：创建一个自定义的Subset wrapper，或者直接使用原始方法
    
    # 方法：为每个subset创建新的数据集，使用原始索引
    from torch.utils.data import Subset
    
    # 获取subset的索引
    train_indices = train_subset_raw.indices
    val_indices = val_subset_raw.indices
    test_indices = test_subset_raw.indices
    
    # 创建带transform的subset
    train_subset = Subset(train_dataset_wrapped, train_indices)
    val_subset = Subset(val_test_dataset_wrapped, val_indices)
    test_subset = Subset(val_test_dataset_wrapped, test_indices)
    
    # ========== 5. 创建DataLoader ==========
    print("\n[5/7] 创建DataLoader...")
    
    batch_size = hyper_args['batch_size']
    # Mac上使用多进程时可能会有问题，建议使用0或1
    # 如果遇到pickle错误，设置为0（单进程）
    num_workers = 0  # Mac上建议使用0（单进程）避免pickle问题
    
    # 设置collate_fn的vocabulary（用于多进程）
    # 减少max_caption_length可以加快训练（但可能影响长句子）
    set_collate_fn_vocab(vocabulary, max_caption_length=25)  # 从30减少到25
    
    # 创建DataLoader（使用模块级别的collate_fn）
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn_wrapper
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn_wrapper
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn_wrapper
    )
    
    print(f"  训练集批次数: {len(train_loader)}")
    print(f"  验证集批次数: {len(val_loader)}")
    print(f"  测试集批次数: {len(test_loader)}")
    
    # ========== 6. 初始化Pipeline和模型 ==========
    print("\n[6/7] 初始化Pipeline和模型...")
    
    pipeline = PipelineIC(vocabulary=vocabulary, device=device)
    
    # 构建模型（按照《Show and Tell》论文配置）
    # 词嵌入维度：512维（论文配置）
    # LSTM隐藏层：512维（论文配置）
    # CNN冻结：论文建议冻结CNN（但实际中微调可能更好）
    freeze_cnn = False  # 设置为True冻结CNN（论文配置），False允许微调（推荐）
    
    pipeline.build_model(
        vocab_size=len(vocabulary),
        embed_dim=512,  # 论文：512维词嵌入
        hidden_dim=512,  # 论文：512维LSTM隐藏层
        num_layers=1,
        dropout=0.5,  # 论文：使用dropout
        freeze_encoder=freeze_cnn  # 论文：冻结CNN权重
    )
    
    if freeze_cnn:
        print("  ✓ CNN编码器已冻结（论文配置）")
    else:
        print("  ✓ CNN编码器允许微调（推荐，效果通常更好）")
    
    # 设置训练参数
    pipeline.set_training_params(
        batch_size=hyper_args['batch_size'],
        n_epoch=hyper_args['epoches'],
        lr=hyper_args['lr'],
        wd=hyper_args['wd']
    )
    
    # ========== 7. 训练和测试 ==========
    print("\n[7/7] 开始训练和测试...")
    
    # 训练
    model_path = os.path.join(folder_args['folder_model'], 'best_model.pth')
    
    if not flag_args.get('load_model_trained', False):
        print("\n" + "=" * 60)
        print("开始训练...")
        print("=" * 60)
        train_results = pipeline.train(train_loader, val_loader)
        
        # 保存模型
        pipeline.save_model(model_path)
        print(f"\n模型已保存到: {model_path}")
    else:
        # 加载已有模型（跳过训练）
        if os.path.exists(model_path):
            print("\n" + "=" * 60)
            print("跳过训练，加载已有模型...")
            print("=" * 60)
            print(f"模型路径: {model_path}")
            pipeline.load_model(model_path, vocab_size=len(vocabulary))
            print("模型加载成功！")
        else:
            print(f"\n错误: 模型文件不存在 {model_path}")
            print("请先训练模型，或检查模型路径是否正确")
            print("如果模型在其他位置，请修改配置文件中的folder_model路径")
            return  # 退出程序
    
    # 测试
    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)
    test_results = pipeline.test(test_loader)
    
    # 保存测试结果
    import json
    results_path = os.path.join(folder_args['folder_test_log'], 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_loss': test_results['test_loss'],
            'bleu_scores': test_results['bleu_scores'],
            'example_samples': test_results['example_samples']
        }, f, indent=2, ensure_ascii=False)
    print(f"\n测试结果已保存到: {results_path}")
    
    print("\n" + "=" * 60)
    print("训练和测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
