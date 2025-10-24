import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config.config import *
from core.PipelineIC import PipelineIC

def main():
    # configuration
    define_dev()
    folder_args, hyper_args, flag_args = read_config()

    # data initialization

    # training and testing pipeline

    # visualization of training and testing results

if __name__ == "__main__":
    main()