# Description: Configuration file for the project

# Directories
IMAGE_DIR = "data/1/Chest-X-Ray/image"
MASK_DIR = "data/1/Chest-X-Ray/mask"
METADATA_CSV = "data/MetaData_cleaned.csv"
DATA_SPLIT_PATH = "data/data_splits"

# Split Ratios
TEST_RATIO = 0.2
VAL_RATIO = 0.25

# Model Parameters
EMBED_DIM = 768
NUM_HEADS = 8
DEPTH = 6
IMG_SIZE = 224
PATCH_SIZE = 16

# Training Parameters
BATCH_SIZE = 64
NUM_EPOCHS = 10
