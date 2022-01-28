import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CovidImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.info_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.df_contains_label = True if 'label' in self.info_df.columns else False

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.info_df.image.iloc[idx])
        image = Image.open(img_name)
        if self.df_contains_label:
            label = 0 if self.info_df.label.iloc[idx] == 'negative' else 1

        if self.transform == 'resize_rotate_crop':
            image = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((240, 240)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.Pad((10, 10)),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(0.8180, 0.1748)
            ])(image)

        else:
            # for validation
            image = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(0.8180, 0.1748)
                ])(image)

        sample = [image, label] if self.df_contains_label else [image, img_name]

        return sample
