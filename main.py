import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=8, depth=6, num_classes_seg=2, num_classes_cls=2, metadata_dim=5):
        super(MultiTaskViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Linear Projection of Patches
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4),
            num_layers=depth
        )

        # Metadata Processing
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)  # Project to the same dimension as image features
        )

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),  # Concatenated image and metadata features
            nn.ReLU(),
            nn.Linear(256, num_classes_cls)
        )

        # Segmentation Head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes_seg, kernel_size=1)
        )

    def forward(self, x, metadata):
        # Patch Embedding
        patches = self.patch_embed(x)  # Shape: [B, embed_dim, H/patch_size, W/patch_size]
        B, C, H, W = patches.shape
        patches = patches.flatten(2).permute(0, 2, 1)  # Shape: [B, num_patches, embed_dim]

        # Add Positional Embedding
        patches += self.pos_embedding

        # Transformer Encoding
        features = self.transformer(patches)  # Shape: [B, num_patches, embed_dim]

        # Metadata Processing
        metadata_features = self.metadata_processor(metadata)  # Shape: [B, embed_dim]

        # Classification Task
        cls_token = features.mean(dim=1)  # Global average pooling over patches
        combined_cls_features = torch.cat([cls_token, metadata_features], dim=1)  # Concatenate image and metadata features
        classification_output = self.classification_head(combined_cls_features)

        # Segmentation Task
        grid_features = features.permute(0, 2, 1).reshape(B, C, H, W)
        metadata_expanded = metadata_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined_seg_features = torch.cat([grid_features, metadata_expanded], dim=1)  # Concatenate image and metadata features
        segmentation_output = self.segmentation_head(combined_seg_features)


        return classification_output, segmentation_output

# Model Initialization
model = MultiTaskViT(
    img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=8, depth=6,
    num_classes_seg=2, num_classes_cls=10, metadata_dim=5
)

# Input Example
image_input = torch.randn(4, 3, 224, 224)  # Batch size 4, RGB images of size 224x224
metadata_input = torch.randn(4, 5)  # Batch size 4, metadata with 5 features (e.g., age, gender, etc.)

classification_output, segmentation_output = model(image_input, metadata_input)

print("Classification output shape:", classification_output.shape)  # [4, num_classes_cls]
print("Segmentation output shape:", segmentation_output.shape)  # [4, num_classes_seg, 224, 224]