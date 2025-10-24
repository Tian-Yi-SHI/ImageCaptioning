import yaml
import math
import torch
from pathlib import Path

def define_dev() -> torch.device:
  if torch.cuda.is_available():
    dev = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
  else:
    dev = torch.device("cpu")
    print("Running on the CPU")
  return dev

def read_config(config_path="./src/config/config_file.yaml") -> tuple:
    '''
    read configuration

    Args:
      config_path: path of configuration file(yaml)
    Return:
      folder_args: storage folder and file
      hyper_args: hyper params
      flag_args: testing and training flags
    '''
    if not Path(config_path).exists():
        raise FileNotFoundError(f"configuration file {config_path} not exist")
    
    # read config file
    with open(config_path, "r", encoding="utf-8") as f:
        try:
          config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
         print(exc)
    
    folder_args = {
        "dataset_name": config.get("dataset_name"),
        "folder_image": config.get("folder_image"),
        "file_caption": config.get("file_caption"),
        "folder_train_log": config.get("folder_train_log"),
        "folder_test_log": config.get("folder_test_log"),
        "folder_model": config.get("folder_model")
    }
    
    hyper_args = {
        "lr": config.get("lr"),
        "wd": config.get("wd"),
        "batch_size": config.get("batch_size"),
        "epoches": config.get("epoches")
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
    