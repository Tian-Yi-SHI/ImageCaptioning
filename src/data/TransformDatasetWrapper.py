from torch.utils.data import Dataset


class TransformDatasetWrapper(Dataset):
    """
    Wrapper for Transforming Images in Dataset
    The Wrapper can be applied to either original dataset or subsets
    
    Situation：
    - Original Dataset with no transform method
    - Specific Transform method designed for subsets
    """
    def __init__(self, dataset, transform):
        """
        Args:
            dataset: original dataset
            transform: transform method
        """
        if transform is None:
            raise RuntimeError("no valid transform method")
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