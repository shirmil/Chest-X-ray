import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from config import DATA_SPLIT_PATH, IMAGE_DIR, MASK_DIR
from config import BATCH_SIZE
from config import EMBED_DIM, NUM_HEADS, DEPTH, IMG_SIZE, PATCH_SIZE
from model import MultiTaskViT
from data_loader import ImageMaskMetadataDataset

#sudo powermetrics --samplers gpu_power -i 1000
def main():
    # Check for MPS availability
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    split_number = 3
    # set metadata path
    if split_number is None:
        split_number = 0
        for split in os.listdir(DATA_SPLIT_PATH):
            if split.startswith("split_"):
                split_num = int(split.split("_")[1])
                if split_num > split_number:
                    split_number = split_num

    print(f"Using split_{split_number}")


    test_metadata_path = os.path.join(DATA_SPLIT_PATH, f"split_{split_number}/test_metadata.csv")
    test_dataset = ImageMaskMetadataDataset(IMAGE_DIR, MASK_DIR, test_metadata_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # load the model from path
    model_path = os.path.join(DATA_SPLIT_PATH, f"split_{split_number}/" + 'model.pth')
    model = MultiTaskViT(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, depth=DEPTH, img_size=IMG_SIZE, patch_size=PATCH_SIZE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Define Loss Functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()

    # Test
    model.eval()
    test_loss_cls = 0.0
    test_loss_seg = 0.0
    correct_cls = 0
    correct_seg = 0
    total_cls = 0
    total_seg = 0

    # Validation
    model.eval()
    test_loss_cls = 0.0
    test_loss_seg = 0.0
    with torch.no_grad():
        for images, metadata, labels_cls, labels_seg in test_loader:
            # Move data to device
            images, metadata, labels_cls, labels_seg = images.to(device), metadata.to(device), labels_cls.to(device), labels_seg.to(device)
            
            outputs_cls, outputs_seg = model(images, metadata)
            loss_cls = criterion_cls(outputs_cls, labels_cls)
            loss_seg = criterion_seg(outputs_seg, labels_seg)

            test_loss_cls += loss_cls.item()
            test_loss_seg += loss_seg.item()
            
            _, predicted_cls = torch.max(outputs_cls, 1)
            _, predicted_seg = torch.max(outputs_seg, 1)
            total_cls += labels_cls.numel()
            correct_cls += (predicted_cls == labels_cls).sum().item()
            
            # Segmentation accuracy
            total_seg += labels_seg.numel()
            correct_seg += (predicted_seg == labels_seg).sum().item()


    print(f"Test Loss_cls: {test_loss_cls / len(test_loader):.4f}, Test Loss_seg: {test_loss_seg / len(test_loader):.4f}")
    print(f"Test Accuracy_cls: {100 * correct_cls / total_cls:.2f}%")
    print(f"Test Accuracy_seg: {100 * correct_seg / total_seg:.2f}%")

if __name__ == '__main__':
    main()

# Output:
# Using device: cpu
# Using split_2
# Test Loss_cls: 0.6931, Test Loss_seg: 0.6931
# Test Accuracy_cls: 50.00%
# Test Accuracy_seg: 50.00%
