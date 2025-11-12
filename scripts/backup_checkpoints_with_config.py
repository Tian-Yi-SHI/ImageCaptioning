"""
备份旧checkpoint并添加训练配置信息
将ResNet18+Adam+大词汇表的checkpoint保存到备份文件夹
"""
import os
import torch
from pathlib import Path
import shutil

def backup_checkpoints_with_config():
    """
    备份旧checkpoint并添加训练配置信息
    """
    # 路径配置
    checkpoint_dir = Path("storage/model/checkpoints")
    backup_dir = Path("storage/model/checkpoints_backup_resnet18_adam")
    
    # 创建备份目录
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # 旧配置信息（ResNet18 + Adam + 大词汇表）
    old_config = {
        'model_architecture': 'ResNet18',
        'optimizer': 'Adam',
        'learning_rate_scheduler': 'ReduceLROnPlateau',
        'vocab_min_freq': 2,
        'vocab_size': 5156,
        'batch_size': 32,
        'data_augmentation': 'RandomHorizontalFlip only',
        'note': 'Old configuration before paper alignment'
    }
    
    # 获取所有checkpoint文件
    checkpoint_files = sorted([
        f for f in checkpoint_dir.glob("checkpoint_epoch_*.pth")
    ], key=lambda x: int(x.stem.split('_')[-1]))
    
    print(f"找到 {len(checkpoint_files)} 个checkpoint文件")
    print(f"备份目录: {backup_dir}")
    print("-" * 80)
    
    # 处理每个checkpoint
    for checkpoint_path in checkpoint_files:
        try:
            print(f"\n处理: {checkpoint_path.name}")
            
            # 加载checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 添加训练配置信息
            checkpoint['training_config'] = old_config
            
            # 添加模型架构信息
            checkpoint['model_architecture'] = 'ResNet18'
            checkpoint['optimizer_type'] = 'Adam'
            checkpoint['vocab_config'] = {
        'min_word_freq': 2,
        'vocab_size': checkpoint.get('vocab_size', 5156)
    }
            
            # 保存到备份目录
            backup_path = backup_dir / checkpoint_path.name
            torch.save(checkpoint, backup_path)
            
            # 也复制原始文件（可选）
            # shutil.copy2(checkpoint_path, backup_dir / f"{checkpoint_path.stem}_original.pth")
            
            print(f"  ✓ 已保存到: {backup_path}")
            print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  - 最佳验证损失: {checkpoint.get('best_val_loss', 'N/A')}")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("✅ 备份完成！")
    print("=" * 80)
    print(f"备份位置: {backup_dir}")
    print(f"备份文件数: {len(checkpoint_files)}")
    print("\n旧checkpoint配置信息:")
    print(f"  - 模型架构: {old_config['model_architecture']}")
    print(f"  - 优化器: {old_config['optimizer']}")
    print(f"  - 词汇表最小词频: {old_config['vocab_min_freq']}")
    print(f"  - 批次大小: {old_config['batch_size']}")
    print("=" * 80)


if __name__ == "__main__":
    backup_checkpoints_with_config()

