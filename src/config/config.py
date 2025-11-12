import os
import torch
import yaml
from pathlib import Path


def define_dev() -> torch.device:
    """
    自动选择最优计算设备，优先级：
    1. NVIDIA CUDA GPU
    2. Apple Silicon MPS（Metal Performance Shaders）
    3. CPU
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("运行设备: NVIDIA GPU (cuda)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("运行设备: Apple Silicon GPU (mps)")
    else:
        dev = torch.device("cpu")
        print("运行设备: CPU")
    return dev

def read_config(config_path=None) -> tuple:
    '''
    read configuration

    Args:
      config_path: path of configuration file(yaml). If None, will try to find it automatically.
    Return:
      folder_args: storage folder and file
      hyper_args: hyper params
      flag_args: testing and training flags
    '''
    # 如果没有提供路径，尝试自动查找
    if config_path is None:
        # 获取当前文件的目录（config.py所在目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 配置文件在同一目录下
        config_path = os.path.join(current_dir, "config_file.yaml")
        
        # 如果还是找不到，尝试从项目根目录查找
        if not os.path.exists(config_path):
            # 尝试从项目根目录查找
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, "src", "config", "config_file.yaml")
    
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"configuration file {config_path} not exist. Please check the path.")
    
    # read config file
    with open(config_path, "r", encoding="utf-8") as f:
        try:
          config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
         print(exc)
    
    # 获取项目根目录（自动向上查找包含 src 目录的路径）
    config_dir = config_path.parent
    project_root = config_dir
    for parent in config_dir.parents:
        if (parent / "src").exists():
            project_root = parent
            break
    
    # 处理相对路径，转换为绝对路径（相对于项目根目录）
    def resolve_path(path, project_root=project_root):
        """将相对路径转换为绝对路径（相对于项目根目录）"""
        if path is None:
            return None
        path = str(path)
        if os.path.isabs(path):
            return path
        # 移除开头的./或./
        path = path.lstrip('./').lstrip('/')
        # 相对于项目根目录
        resolved = os.path.join(project_root, path)
        return os.path.normpath(resolved)  # 规范化路径
    
    folder_args = {
        "dataset_name": config.get("dataset_name"),
        "folder_image": resolve_path(config.get("folder_image")),
        "file_caption": resolve_path(config.get("file_caption")),
        "folder_train_log": resolve_path(config.get("folder_train_log")),
        "folder_test_log": resolve_path(config.get("folder_test_log")),
        "folder_model": resolve_path(config.get("folder_model"))
    }
    
    hyper_args = {
        "lr": config.get("lr"),
        "wd": config.get("wd"),
        "batch_size": config.get("batch_size"),
        "epoches": config.get("epoches"),
        "early_stop_patience": config.get("early_stop_patience", 0)  # 早停patience，默认0（禁用）
    }
    
    flag_args = {
        "load_model_trained": config.get("load_model_trained")
    }
    
    return folder_args, hyper_args, flag_args

if __name__ == "__main__":
    define_dev()
    folder_args, hyper_args, flag_args = read_config()
    
    print(folder_args)
    print(hyper_args)
    print(flag_args)
    