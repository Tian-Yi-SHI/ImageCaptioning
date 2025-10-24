import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import Dataset, Subset

class TransformDatasetWrapper(Dataset):
    """
    Wrapper to Transform images in Dataset
    
    Situations：
    - Original Dataset with no transform method or need to be covered
    - set specific transform method for subset
    """
    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset: original dataset
            transform: transform method
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image, captions, image_name = sample
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, captions, image_name