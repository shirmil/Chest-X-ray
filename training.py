import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import DATA_SPLIT_PATH, IMAGE_DIR, MASK_DIR
from config import BATCH_SIZE, NUM_EPOCHS
from config import EMBED_DIM, NUM_HEADS, DEPTH, IMG_SIZE, PATCH_SIZE
from model import MultiTaskViT

class ToLongTensor:
    def __call__(self, pic):
        return pic.clone().detach().to(torch.long)

class ImageMaskMetadataDataset(Dataset):
    def __init__(self, image_dir, mask_dir, metadata_csv, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.metadata = pd.read_csv(metadata_csv)
        self.image_transform = image_transform
        self.mask_transform = mask_transform



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

        # Convert metadata to tensor
        metadata = torch.tensor(metadata, dtype=torch.float32)

        # (images, metadata, labels_cls, labels_seg)
        return image, metadata, label_cls, mask

# Example usage
image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to the desired size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for single-channel grayscale
])

mask_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to the desired size
    transforms.ToTensor(),  # Convert to tensor
    ToLongTensor()  # Convert to long tensor for CrossEntropyLoss
])

split_number = None

# set metadata path
if split_number is None:
    split_number = 0
    for split in os.listdir(DATA_SPLIT_PATH):
        if split.startswith("split_"):
            split_num = int(split.split("_")[1])
            if split_num > split_number:
                split_number = split_num

print(f"Using split_{split_number}")

train_metadata_path = os.path.join(DATA_SPLIT_PATH, f"split_{split_number}/train_metadata.csv")
val_metadata_path = os.path.join(DATA_SPLIT_PATH, f"split_{split_number}/val_metadata.csv")
test_metadata_path = os.path.join(DATA_SPLIT_PATH, f"split_{split_number}/test_metadata.csv")

# Create datasets
train_dataset = ImageMaskMetadataDataset(IMAGE_DIR, MASK_DIR, train_metadata_path, image_transform, mask_transform)
val_dataset = ImageMaskMetadataDataset(IMAGE_DIR, MASK_DIR, val_metadata_path, image_transform, mask_transform)
test_dataset = ImageMaskMetadataDataset(IMAGE_DIR, MASK_DIR, test_metadata_path, image_transform, mask_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Initialization
model = MultiTaskViT(
    img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=1, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, depth=DEPTH,
    num_classes_seg=2, num_classes_cls=2, metadata_dim=4
)

# Define Loss Functions
criterion_cls = nn.CrossEntropyLoss()
criterion_seg = nn.CrossEntropyLoss()

# Define Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Validation Loop

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss_cls = 0.0
    running_loss_seg = 0.0
    
    for i, (images, metadata, labels_cls, real_mask_seg) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs_cls, outputs_seg = model(images, metadata)

        # Compute losses
        loss_cls = criterion_cls(outputs_cls, labels_cls.long())
        loss_seg = criterion_seg(outputs_seg, real_mask_seg)
        
        # Backward pass and optimization
        loss = loss_cls + loss_seg
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        running_loss_cls += loss_cls.item()
        running_loss_seg += loss_seg.item()
        
        if i % 10 == 9:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_loader)}], '
                  f'Loss_cls: {running_loss_cls / 10:.4f}, Loss_seg: {running_loss_seg / 10:.4f}')
            running_loss_cls = 0.0
            running_loss_seg = 0.0
    
    # Validation
    model.eval()
    val_loss_cls = 0.0
    val_loss_seg = 0.0
    with torch.no_grad():
        for images, metadata, labels_cls, labels_seg in val_loader:
            outputs_cls, outputs_seg = model(images, metadata)
            loss_cls = criterion_cls(outputs_cls, labels_cls.long())
            loss_seg = criterion_seg(outputs_seg, labels_seg)
            val_loss_cls += loss_cls.item()
            val_loss_seg += loss_seg.item()
    
    val_loss_cls /= len(val_loader)
    val_loss_seg /= len(val_loader)
    print(f'Validation Loss_cls: {val_loss_cls:.4f}, Validation Loss_seg: {val_loss_seg:.4f}')

    # Save the model after each epoch
    # Path to save the model
    model_save_path = os.path.join(DATA_SPLIT_PATH, 'model.pth')

    torch.save(model.state_dict(), model_save_path)

print('Training finished.')