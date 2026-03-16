import torch
import sys
import os
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dinov2.models.vision_transformer import vit_base
from dinov2.data.transforms import make_normalize_transform

device = torch.device('cuda')


checkpoint = torch.load("../outputs/eval/training_1499/teacher_checkpoint.pth", map_location="cpu", weights_only=False)['teacher']
# checkpoint_dict = checkpoint
teacher_backbone_checkpoint = {}
for key in checkpoint.keys():
    if 'backbone' in key:
        weight_key = key[9:]
        teacher_backbone_checkpoint[weight_key] = checkpoint[key]

# print(teacher_backbone_checkpoint['pos_embed'])
# print(teacher_backbone_checkpoint['pos_embed'].shape)
dino_pos_embed = teacher_backbone_checkpoint['pos_embed']
cls_token_pos_embed = dino_pos_embed[:, :1, :] # Shape: (1, 1, 768)
patch_pos_embed_2d = dino_pos_embed[:, 1:, :] # Shape: (1, 1369, 768)

# Calculate original grid dimensions (37x37)
orig_grid_h = orig_grid_w = int(patch_pos_embed_2d.shape[1]**0.5) # Should be 37

# Reshape from (1, H*W, C) to (1, C, H, W)
patch_pos_embed_2d = patch_pos_embed_2d.permute(0, 2, 1).reshape(
    1, -1, orig_grid_h, orig_grid_w
) # Shape: (1, 768, 37, 37)

target_grid_h = target_grid_w = 64 # For 1024x1024 image with 16x16 patches

interpolated_patch_pos_embed = F.interpolate(
    patch_pos_embed_2d,
    size=(target_grid_h, target_grid_w),
    mode='bicubic', # Typically bicubic for smoothness
    align_corners=False # Standard practice for bicubic interpolation
) # Shape: (1, 768, 64, 64)
print(interpolated_patch_pos_embed.shape)

# Flatten back to (1, H_new*W_new, C)
interpolated_patch_pos_embed = interpolated_patch_pos_embed.flatten(2).permute(0, 2, 1) # Shape: (1, 4096, 768)
print(interpolated_patch_pos_embed.shape)

# Concatenate the CLS token back (if SAM's pos_embed expects it)
interpolated_pos_embed_with_cls = torch.cat([cls_token_pos_embed, interpolated_patch_pos_embed], dim=1)
print(interpolated_pos_embed_with_cls.shape)

teacher_backbone_checkpoint['pos_embed'] = interpolated_pos_embed_with_cls

vit_model = vit_base(
    img_size=896, #896
    patch_size=14,
    block_chunks=0,
    num_register_tokens=4,
    init_values=1e-5
).to(device)
vit_model.load_state_dict(teacher_backbone_checkpoint)

# print(vit_model.state_dict().keys())

img = Image.open(f'../../../DINO_Dataset_1024/train/fluorescent/{os.listdir("../../../DINO_Dataset_1024/train/fluorescent/")[0]}').convert('RGB')
original_image_np = np.array(img)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        make_normalize_transform(), # need to edit mean and std
        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
    ]
)
img = transform(img).unsqueeze(0).to(device)

dino_out = vit_model(img)
print(dino_out)
print(dino_out['x_norm_patchtokens'].shape)
patch_features = dino_out['x_norm_patchtokens']

pca = PCA(n_components=3)
pca_features = pca.fit_transform(patch_features[0].detach().cpu())


pca_features_normalized = np.zeros_like(pca_features)
for i in range(pca_features.shape[1]):
    min_val = pca_features[:, i].min()
    max_val = pca_features[:, i].max()
    if max_val - min_val > 0:
        pca_features_normalized[:, i] = (pca_features[:, i] - min_val) / (max_val - min_val)
    else:
        pca_features_normalized[:, i] = 0.5 # Handle case where component has no variance

pca_image = pca_features_normalized.reshape(64, 64, 3) # Reshape to a grid

# 5. Upscale and Visualize
# Upscale the PCA image to the original cropped image dimensions for visualization
pca_image_upscaled = cv2.resize(pca_image, (896, 896), interpolation=cv2.INTER_LINEAR)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_image_np)
plt.title("Original Image (Cropped)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pca_image_upscaled)
plt.title(f"DINOv2 PCA Visualization")
plt.axis('off')

plt.tight_layout()
plt.savefig(f'output_3.png')
plt.close()

# Print explained variance ratio (how much variance each PC captures)
print("\nExplained variance ratio by principal components:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ratio:.4f}")
print(f"  Total explained by 3 PCs: {pca.explained_variance_ratio_.sum():.4f}")