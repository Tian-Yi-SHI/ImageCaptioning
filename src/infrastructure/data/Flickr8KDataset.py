import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class Flickr8KDataset(Dataset):
    """
    Flickr8K Dataset wwithout transform method
    return images and captions
    """
    def __init__(self, folder_path_image, file_path_caption):
        """
        Args:
            folder_path_image: image folder path
            file_path_caption: caption file path
            transform: image transform method
        """
        self.folder_path_image = folder_path_image
        
        self.captions_df = pd.read_csv(file_path_caption)
        
        # available image names
        if not os.path.isdir(folder_path_image):
            raise NotADirectoryError(f"图片文件夹路径无效：{folder_path_image}")
        actual_image_names = set(os.listdir(folder_path_image))
        
        # filter caption file
        self.captions_df['image_exists'] = self.captions_df['image'].isin(actual_image_names)
        filtered_df = self.captions_df[self.captions_df['image_exists']].copy()
        
        # available images
        self.image_captions = filtered_df.groupby('image')['caption'].apply(list).to_dict()
        self.image_names = list(self.image_captions.keys())
        
        # check
        if not self.image_names:
            raise ValueError("no image available")

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
        
        return image, captions, image_name
    
from torchvision import transforms

if __name__ == "__main__":
    # initialization
    dataset = Flickr8KDataset(
        folder_path_image="./storage/dataset/archive/images",
        file_path_caption="./storage/dataset/captions/captions.txt",
    )

    # test
    for i in range(len(dataset)):
        image, captions, image_name = dataset[i]
        print(i)
    image, captions, image_name = dataset[0]
    print(f"image name: {image_name}")
    print(f"shape: {image.shape}")
    print(f"corresponding captions: {len(captions)} in total")
    for i, cap in enumerate(captions, 1):
        print(f"  caption {i}: {cap}")