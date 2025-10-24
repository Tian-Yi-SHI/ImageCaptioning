import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class Flickr8KDataset(Dataset):
    """
    Flickr8K Dataset
    return images and captions
    """
    def __init__(self, folder_path_image, file_path_caption, transform=None):
        """
        Args:
            folder_path_image: image folder path
            file_path_caption: caption file path
            transform: image transform method
        """
        self.folder_path_image = folder_path_image
        self.transform = transform
        
        # read caption file
        self.captions_df = pd.read_csv(file_path_caption)
        
        # group captions by image name
        # format：{image_name: [caption1, caption2, ..., caption5]}
        self.image_captions = self.captions_df.groupby('image')['caption'].apply(list).to_dict()
        
        # image name
        self.image_names = list(self.image_captions.keys())

    def __len__(self):
        """return the total num of images"""
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        return image and related captions according to given index
        
        Returns:
            image: image tensor after transform
            captions: list of captions
            image_name: image name
        """
        # get image name and corresponding captions
        image_name = self.image_names[idx]
        captions = self.image_captions[image_name]
        
        # image path
        image_path = os.path.join(self.folder_path_image, image_name)
        
        # read image
        image = Image.open(image_path).convert('RGB')
        
        # transform
        if self.transform is not None:
            image = self.transform(image)
        
        return image, captions, image_name
    
# from config.config import *
from torchvision import transforms

# transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# initialization
dataset = Flickr8KDataset(
    folder_path_image="./storage/dataset/archive/images",
    file_path_caption="./storage/dataset/captions/captions.txt",
    transform=transform
)

# test
image, captions, image_name = dataset[0]
print(f"image name: {image_name}")
print(f"shape: {image.shape}")
print(f"corresponding captions: {len(captions)} in total")
for i, cap in enumerate(captions, 1):
    print(f"  caption {i}: {cap}")