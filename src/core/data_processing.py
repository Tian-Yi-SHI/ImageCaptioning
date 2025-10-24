import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from infrastructure.data.Flickr8KDataset import Flickr8KDataset
from infrastructure.data.TransformDatasetWrapper import TransformDatasetWrapper

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    split dataset and return result subsets
    
    Args:
        dataset: Flickr8KDataset
        train_ratio/val_ratio/test_ratio: subset ratio
        random_seed: random seed
    Returns:
        train_subset, val_subset, test_subset: result subsets
    """
    # check
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "ratios must sum up to 1.0"
    
    # indices
    total_size = len(dataset)
    all_indices = np.arange(total_size)
    
    # indice split
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=val_ratio + test_ratio,
        random_state=random_seed,
        shuffle=True
    )
    val_size = int(val_ratio / (val_ratio + test_ratio) * len(temp_indices))
    val_indices = temp_indices[:val_size]
    test_indices = temp_indices[val_size:]
    
    # create subset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    print(f"finish dataset split")
    print(f"training set：{len(train_subset)} images")
    print(f"validation set：{len(val_subset)} images")
    print(f"testing set：{len(test_subset)} images")
    
    return train_subset, val_subset, test_subset

def get_dataloaders(train_subset, val_subset, test_subset, batch_size=32, num_workers=8):
    '''
    generate dataloaders from subsets

    Warning:
        Transform is a must before creating dataloader(iterator in torch only support tensor operation, not Images)
    '''
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True
    )

    print("finish dataloader creation")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # path
    IMAGE_DIR = "./storage/dataset/archive/images"
    CAPTION_PATH = "./storage/dataset/captions/captions.txt"
    
    # transforms
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
    
    # load dataset
    dataset = Flickr8KDataset(
        folder_path_image=IMAGE_DIR,
        file_path_caption=CAPTION_PATH,
    )
    dataset = TransformDatasetWrapper(dataset, val_test_transform)
    
    # split dataset
    train_subset, val_subset, test_subset = split_dataset(
        dataset=dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # subset transforms
    # train_subset = TransformDatasetWrapper(train_subset, train_transform)
    # val_subset = TransformDatasetWrapper(val_subset, val_test_transform)
    # test_subset = TransformDatasetWrapper(test_subset, val_test_transform)
    
    # create DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(
        train_subset, val_subset, test_subset, batch_size=32
    )
    
    # test
    for images, captions, image_names in train_loader:
        print(f"training set: shape {images.shape}, caption: {captions[0][0]}")
        break
    
    for images, captions, image_names in val_loader:
        print(f"validation set: shape {images.shape}, caption: {captions[0][0]}")
        break