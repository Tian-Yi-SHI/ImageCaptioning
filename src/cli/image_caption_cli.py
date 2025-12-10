"""
图像描述生成命令行工具
用于训练和测试模型
"""
import sys
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径，以便正确导入模块
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# 添加src目录到Python路径
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# 使用绝对导入，从src开始
from src.config.config import define_dev, read_config
from src.pipeline.PipelineIC import PipelineIC
from src.pipeline.data_processing import split_dataset, get_dataloaders
from src.utils.vocabulary import Vocabulary
from src.utils.collate_fn import collate_fn
from src.data.Flickr8KDataset import Flickr8KDataset
from src.data.TransformDatasetWrapper import TransformDatasetWrapper


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
    
    # 优先使用数据集中缓存的caption，避免重复加载图像
    if hasattr(dataset, 'get_all_captions'):
        for captions in dataset.get_all_captions().values():
            captions_list.append(captions)
    else:
        # 回退方案：逐条访问数据集（可能较慢）
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


def run_quick_sample_check(pipeline, test_subset, num_workers, sample_count, checkpoint_path, model_path, vocab_size):
    """
    在继续训练前对少量样本进行快速检测，输出真实与生成描述
    """
    sample_count = max(1, sample_count)
    if test_subset is None:
        print("\n>>> 快速检测: 无测试数据，跳过")
        return
    
    available = len(test_subset) if hasattr(test_subset, "__len__") else 0
    if available == 0:
        print("\n>>> 快速检测: 测试数据为空，跳过")
        return
    
    actual_count = min(sample_count, available)
    
    # 构建快速子集
    if hasattr(test_subset, "indices"):
        target_indices = test_subset.indices[:actual_count]
        base_dataset = test_subset.dataset
        quick_subset = Subset(base_dataset, target_indices)
    else:
        quick_subset = Subset(test_subset, list(range(actual_count)))
    
    quick_loader = DataLoader(
        quick_subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn_wrapper
    )
    
    # 加载模型权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n>>> 快速检测: 加载checkpoint {checkpoint_path}")
        pipeline.load_checkpoint_model(checkpoint_path)
    elif model_path and os.path.exists(model_path):
        print(f"\n>>> 快速检测: 加载已训练模型 {model_path}")
        pipeline.load_model(model_path, vocab_size=vocab_size)
    else:
        print("\n>>> 快速检测: 未找到可用模型，跳过")
        return
    
    print(f"\n>>> 正在对 {actual_count} 张图片进行快速检测，输出真实与生成描述 ...")
    try:
        pipeline.test(quick_loader)
    except Exception as exc:
        print(f">>> 快速检测失败: {exc}")
    else:
        print("\n>>> 快速检测完成\n")


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
    
    # 获取模型类型（默认使用transformer）
    model_type = flag_args.get('model_type', 'transformer')
    
    # 构建模型（按照《Show and Tell》论文配置）
    # 词嵌入维度：512维（论文配置）
    # LSTM隐藏层：512维（论文配置）
    # CNN冻结：论文建议冻结CNN（但实际中微调可能更好）
    # 默认冻结CNN，先保证特征稳定；如需微调可在此改为False或提供配置开关
    freeze_cnn = True
    
    if model_type == 'transformer':
        # Transformer模型配置
        pipeline.build_model(
            vocab_size=len(vocabulary),
            model_type='transformer',
            d_model=512,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            dropout=0.1,
            pos_dim=256,
            feature_dim=256,
            freeze_encoder=freeze_cnn,
            threshold_mode='adaptive'
        )
    else:  # 'cnn_lstm'
        # CNN-LSTM模型配置（按照《Show and Tell》论文）
        pipeline.build_model(
            vocab_size=len(vocabulary),
            model_type='cnn_lstm',
            embed_dim=512,  # 论文：512维词嵌入
            hidden_dim=512,  # 论文：512维LSTM隐藏层
            num_layers=1,
            dropout=0.5,  # 论文：使用dropout
            freeze_encoder=freeze_cnn  # 论文：冻结CNN权重
        )
    
    if freeze_cnn:
        print("  ✓ CNN编码器已冻结，优先保护预训练特征")
    else:
        print("  ✓ CNN编码器允许微调（需确保较小学习率）")
    
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
    checkpoint_dir = os.path.join(folder_args['folder_model'], 'checkpoints')
    
    # 检查是否有checkpoint可以恢复训练
    checkpoint_path = None
    if os.path.exists(checkpoint_dir):
        # 查找最新的checkpoint文件
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        if checkpoint_files:
            # 按epoch编号排序，获取最新的
            checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"\n发现checkpoint: {checkpoint_path}")
            print("可以选择从checkpoint恢复训练，或从头开始训练")
            
            # 在继续训练前进行快速抽样检测
            run_quick_sample_check(
                pipeline=pipeline,
                test_subset=test_subset,
                num_workers=num_workers,
                sample_count=10,
                checkpoint_path=checkpoint_path,
                model_path=model_path,
                vocab_size=len(vocabulary)
            )
    
    if not flag_args.get('load_model_trained', False):
        print("\n" + "=" * 60)
        print("开始训练...")
        print("=" * 60)
        
        # 如果找到了checkpoint，检查模型类型是否一致
        if checkpoint_path:
            print(f"\n检测到checkpoint: {checkpoint_path}")
            
            # 检查checkpoint中的模型类型
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                checkpoint_model_type = checkpoint.get('model_type', 'cnn_lstm')
                
                if checkpoint_model_type != model_type:
                    print(f"\n⚠️  警告：配置的模型类型 ({model_type}) 与checkpoint中的模型类型 ({checkpoint_model_type}) 不一致！")
                    print(f"   如果继续，将从checkpoint恢复 {checkpoint_model_type} 模型，而不是配置的 {model_type} 模型。")
                    print(f"   如果要使用 {model_type} 模型，请：")
                    print(f"   1. 删除checkpoint目录: {checkpoint_dir}")
                    print(f"   2. 或者将配置文件中的 model_type 改为: {checkpoint_model_type}")
                    print(f"\n   当前将使用checkpoint中的模型类型: {checkpoint_model_type}")
                    # 更新model_type以匹配checkpoint
                    model_type = checkpoint_model_type
                else:
                    print(f"   模型类型匹配: {model_type}")
                    print("   将自动从checkpoint恢复训练")
            except Exception as e:
                print(f"   无法读取checkpoint信息: {e}")
                print("   将从头开始训练")
                checkpoint_path = None
        
        # 开始训练，传入checkpoint_path和checkpoint_dir
        train_results = pipeline.train(
            train_loader, 
            val_loader, 
            checkpoint_path=checkpoint_path, 
            checkpoint_dir=checkpoint_dir
        )
        
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
    
    # 检查是否要测试特定的checkpoint（第25个epoch）
    test_checkpoint_epoch = None
    checkpoint_25_path = None
    if os.path.exists(checkpoint_dir):
        checkpoint_25_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_25.pth')
        if os.path.exists(checkpoint_25_path):
            print(f"\n发现第25个epoch的checkpoint，将测试该checkpoint")
            test_checkpoint_epoch = 25
            # 加载checkpoint模型（会覆盖之前加载的模型）
            pipeline.load_checkpoint_model(checkpoint_25_path)
    
    # 执行测试
    test_results = pipeline.test(test_loader)
    
    # 保存测试结果
    import json
    if test_checkpoint_epoch is not None:
        # 保存checkpoint测试结果
        results_path = os.path.join(folder_args['folder_test_log'], f'test_results_epoch_{test_checkpoint_epoch}.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'checkpoint_epoch': test_checkpoint_epoch,
                'test_loss': test_results['test_loss'],
                'bleu_scores': test_results['bleu_scores'],
                'example_samples': test_results['example_samples']
            }, f, indent=2, ensure_ascii=False)
        print(f"\n测试结果已保存到: {results_path}")
    else:
        # 保存正常测试结果
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
