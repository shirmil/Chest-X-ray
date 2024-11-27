import kagglehub
import os

# Ensure the 'data' folder exists
os.makedirs('data', exist_ok=True)

# Download the dataset to the 'data' folder
path = kagglehub.dataset_download("iamtapendu/chest-x-ray-lungs-segmentation", force_download=True)

# Move the downloaded dataset to the 'data' folder if necessary
os.rename(path, os.path.join('data', os.path.basename(path)))

# Print the path to verify
print("Dataset downloaded to:", os.path.join('data', os.path.basename(path)))