import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
from model import MultiTaskViT

# Assuming ImageMaskMetadataDataset is defined elsewhere
# from your_dataset_module import ImageMaskMetadataDataset

# Define the path to the metadata CSV file
METADATA_CSV = 'data/MetaData.csv'
DATA_SPLIT_PATH = 'data/data_split'
IMAGE_DIR = 'path/to/image_dir'
MASK_DIR = 'path/to/mask_dir'
BATCH_SIZE = 32
NUM_EPOCHS = 10
IMG_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768
NUM_HEADS = 8
DEPTH = 6

# Check for MPS availability
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to the desired size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

mask_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to the desired size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Lambda(lambda x: x.long())  # Convert to long tensor for CrossEntropyLoss
])

# Create datasets
train_dataset = ImageMaskMetadataDataset(IMAGE_DIR, MASK_DIR, METADATA_CSV, image_transform=image_transform, mask_transform=mask_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Initialization
model = MultiTaskViT(
    img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=3, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, depth=DEPTH,
    num_classes_seg=2, num_classes_cls=2, metadata_dim=4
).to(device)

# Define Loss Functions
criterion_cls = nn.CrossEntropyLoss()
criterion_seg = nn.CrossEntropyLoss()

# Define Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Path to save the model
model_save_path = os.path.join(DATA_SPLIT_PATH, 'model.pth')

# Training and Validation Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss_cls = 0.0
    running_loss_seg = 0.0
    
    for i, (images, metadata, labels_cls, real_mask_seg) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Move data to device
        images, metadata, labels_cls, real_mask_seg = images.to(device), metadata.to(device), labels_cls.to(device), real_mask_seg.to(device)
        
        # Forward pass
        outputs_cls, outputs_seg = model(images, metadata)
        
        # Convert labels_cls to Long (int64)
        labels_cls = labels_cls.long()
        
        # Ensure real_mask_seg is a 3D tensor and convert to Long
        if real_mask_seg.dim() == 4:
            real_mask_seg = real_mask_seg.squeeze(1)
        real_mask_seg = real_mask_seg.long()
        
        # Compute losses
        loss_cls = criterion_cls(outputs_cls, labels_cls)
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
            # Move data to device
            images, metadata, labels_cls, labels_seg = images.to(device), metadata.to(device), labels_cls.to(device), labels_seg.to(device)
            
            outputs_cls, outputs_seg = model(images, metadata)
            loss_cls = criterion_cls(outputs_cls, labels_cls.long())
            loss_seg = criterion_seg(outputs_seg, labels_seg)
            val_loss_cls += loss_cls.item()
            val_loss_seg += loss_seg.item()
    
    val_loss_cls /= len(val_loader)
    val_loss_seg /= len(val_loader)
    print(f'Validation Loss_cls: {val_loss_cls:.4f}, Validation Loss_seg: {val_loss_seg:.4f}')
    
    # Save the model after each epoch
    torch.save(model.state_dict(), model_save_path)

print('Training finished.')