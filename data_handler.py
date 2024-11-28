import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from config import METADATA_CSV, DATA_SPLIT_PATH
from config import TEST_RATIO, VAL_RATIO


# Load metadata
metadata = pd.read_csv(METADATA_CSV)

# Split into training, validation, and test sets
train_metadata, test_metadata = train_test_split(metadata, test_size=TEST_RATIO, random_state=42)
train_metadata, val_metadata = train_test_split(train_metadata, test_size=VAL_RATIO, random_state=42)  # 0.25 x 0.8 = 0.2


# Check the next available number for split_# & Create the new split directory
split_number = 1
while os.path.exists(os.path.join(DATA_SPLIT_PATH, f"split_{split_number}")):
    split_number += 1
new_split_dir = os.path.join(DATA_SPLIT_PATH, f"split_{split_number}")
os.makedirs(new_split_dir)

# create paths
train_metadata_path = os.path.join(new_split_dir, "train_metadata.csv")
val_metadata_path = os.path.join(new_split_dir, "val_metadata.csv")
test_metadata_path = os.path.join(new_split_dir, "test_metadata.csv")

# Save split CSVs for reproducibility (optional)
train_metadata.to_csv(train_metadata_path, index=False)
val_metadata.to_csv(val_metadata_path, index=False)
test_metadata.to_csv(test_metadata_path, index=False)

