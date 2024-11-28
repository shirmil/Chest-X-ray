import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768, num_heads=8, depth=6, num_classes_seg=2, num_classes_cls=2, metadata_dim=4):
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
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True),
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

        # Segmentation Decoder with Skip Connections
        self.upconv1 = nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2)  # Upsample by 2x
        self.conv1 = nn.Conv2d(256 + embed_dim * 2, 128, kernel_size=3, padding=1)  # Add skip and metadata
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Upsample again
        self.conv2 = nn.Conv2d(64 + embed_dim * 2, 64, kernel_size=3, padding=1)  # Add skip and metadata
        self.final_conv = nn.Conv2d(64, num_classes_seg, kernel_size=1)  # Output segmentation map


    def forward(self, x, metadata):
        # Patch Embedding
        patches = self.patch_embed(x)  # Shape: [B, embed_dim, H/patch_size, W/patch_size]
        B, C, H, W = patches.shape
        skip1 = patches  # Save for skip connection
        patches = patches.flatten(2).permute(0, 2, 1)  # Shape: [B, num_patches, embed_dim]

        # Add Positional Embedding
        patches += self.pos_embedding

        # Transformer Encoding
        features = self.transformer(patches)  # Shape: [B, num_patches, embed_dim]

        # Metadata Processing
        metadata_features = self.metadata_processor(metadata)  # Shape: [B, embed_dim]
        metadata_features = metadata_features.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        metadata_features = metadata_features.expand(-1, -1, H, W)  # Broadcast to match spatial dimensions

        # Classification Task
        cls_token = features.mean(dim=1)  # Global average pooling over patches
        combined_cls_features = torch.cat([cls_token, metadata_features[:, :, 0, 0]], dim=1)  # Combine metadata
        classification_output = self.classification_head(combined_cls_features)

        # Segmentation Task
        grid_features = features.permute(0, 2, 1).reshape(B, C, H, W)
        skip2 = grid_features  # Save for another skip connection

        # Segmentation Task
        x = self.upconv1(grid_features)  # Upsample
        skip2 = F.interpolate(skip2, size=x.shape[2:], mode='bilinear', align_corners=False)  # Upsample skip2 to match x
        metadata_features2 = F.interpolate(metadata_features, size=x.shape[2:], mode='bilinear', align_corners=False)  # Upsample metadata_features to match x 
        x = torch.cat([x, skip2, metadata_features2], dim=1)  # Add skip connection and metadata
        x = self.conv1(x)

        x = self.upconv2(x)  # Upsample again
        skip1 = F.interpolate(skip1, size=x.shape[2:], mode='bilinear', align_corners=False)  # Upsample skip2 to match x
        metadata_features1 = F.interpolate(metadata_features, size=x.shape[2:], mode='bilinear', align_corners=False)  # Upsample metadata_features to match x 
        x = torch.cat([x, skip1, metadata_features1], dim=1)  # Add another skip connection and metadata
        x = self.conv2(x)

        segmentation_output = F.interpolate(self.final_conv(x), size=(x.shape[2] * 4, x.shape[3] * 4), mode='bilinear', align_corners=False)  # Final segmentation map

        return classification_output, segmentation_output

# Model Initialization
# model = MultiTaskViT(
#     img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=8, depth=6,
#     num_classes_seg=2, num_classes_cls=2, metadata_dim=4
# )

# Input Example
# image_input = torch.randn(4, 3, 224, 224)  # Batch size 4, RGB images of size 224x224
# metadata_input = torch.randn(4, 4)  # Batch size 4, metadata with 5 features (e.g., age, gender, etc.)

# classification_output, segmentation_output = model(image_input, metadata_input)

# print("Classification output shape:", classification_output.shape)  # [4, num_classes_cls]
# print("Segmentation output shape:", segmentation_output.shape)  # [4, num_classes_seg, 224, 224]