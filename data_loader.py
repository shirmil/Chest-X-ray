import os
import pandas as pd
from config import IMG_SIZE
import os
import pandas as pd
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms

class ToLongTensor:
    def __call__(self, pic):
        return pic.clone().detach().to(torch.long)

class ImageMaskMetadataDataset(Dataset):
    def __init__(self, image_dir, mask_dir, metadata_csv):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.metadata = pd.read_csv(metadata_csv)

        self.image_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to the desired size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for single-channel grayscale
        ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to the desired size
                transforms.ToTensor(),  # Convert to tensor
                ToLongTensor()  # Convert to long tensor for CrossEntropyLoss
            ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get the image file name and corresponding mask
        row = self.metadata.iloc[idx]
        file_name = str(int(row['id'])) + '.png'
        image_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)

        # Load image and mask
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are grayscale

        # Load metadata
        label_cls = row['ptb']
        metadata = row.drop(['id', 'ptb']).values

        # Apply transformations
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Remove channel dimension from mask
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        # Convert metadata to tensor and ensure float32
        metadata = torch.tensor(metadata, dtype=torch.float32)

        # Convert label_cls to long
        label_cls = torch.tensor(label_cls, dtype=torch.long)

        # Convert mask to long
        mask = mask.long()

        # (images, metadata, labels_cls, labels_seg)
        return image, metadata, label_cls, mask