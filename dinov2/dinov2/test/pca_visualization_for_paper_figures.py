import torch
from torchvision import transforms as T
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2 # For resizing and interpolation
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import DinoVisionTransformer directly for un-sharded instantiation
from dinov2.dinov2.models.vision_transformer import vit_small, vit_large, vit_base
from dinov2.dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.dinov2.models import build_model_from_cfg
from dinov2.dinov2.data.transforms import CLAHETransform, make_normalize_transform


def run_pca_visualization_DINOCell(model, img_path, fig_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_size = 8
    global_crop_size = 518 # Or 518, matching your training config


    if 'decoder_SAM' in fig_name:
        teacher_backbone_checkpoint = model.dino_encoder.dino_encoder.state_dict()
    elif 'direct_mask_prediction_independent_encoding' in fig_name:
        teacher_backbone_checkpoint = model.direct_mask_prediction_image_encoder.dino_encoder.state_dict()
    else:
        teacher_backbone_checkpoint = model.encoder.dino_encoder.state_dict()

    num_patches = 896 // patch_size
    
    vit_model = vit_base(
        img_size=896, #896
        patch_size=patch_size,
        block_chunks=0,
        num_register_tokens=4,
        init_values=1e-5
    ).to(device)
    vit_model.load_state_dict(teacher_backbone_checkpoint)

    # img = Image.open(f'../../DINO_Dataset_1024/train/pc_bf/{os.listdir("../../DINO_Dataset_1024/train/pc_bf/")[0]}').convert('RGB')
    # img = Image.open('../../DINO_Dataset_1024/train/pc_bf/Cellpainting_Gallery_cpg0024_cpg0024-bortezomib_source_4_images_2021_08_23_Batch12_images_BR00126734__2021-09-08T09_18_47-Measurement2_Imagesr03c13f02p01-ch6sk1fk1fl1.tiff')
    img = Image.open(img_path).convert("RGB")
    # img = Image.open('../../Dataset/test_image_for_pca_output.png').convert('RGB')
    original_image_np = np.array(img)
    transform = T.Compose(
        [
            T.ToTensor(),
            make_normalize_transform(), # need to edit mean and std
            T.Resize(896, interpolation=T.InterpolationMode.BICUBIC),
        ]
    )
    img = transform(img).unsqueeze(0).to(device)
    if img.shape[2] != img.shape[3]:
        img = img[:, :, :896, :896]

    # Normalize to 0–1
    save_img = img[0].detach().cpu().numpy()
    save_img_min = save_img.min()
    save_img_max = save_img.max()
    save_img_norm = (save_img - save_img_min) / (save_img_max - save_img_min + 1e-8)

    # Convert to 0–255 uint8
    save_img = (save_img_norm * 255).astype(np.uint8).transpose(1, 2, 0)
    save_img = Image.fromarray(save_img)
    save_img.save(f'{fig_name.split(".")[0]}_img_crop.png')

    # if img.shape[2] != img.shape[3]:
    #     img = img[:, :, :img.shape[3], :img.shape[3]] # Crop to square if not already

    dino_out = vit_model(img)
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

    pca_image = pca_features_normalized.reshape(num_patches, num_patches, 3) # Reshape to a grid

    # 5. Upscale and Visualize
    # Upscale the PCA image to the original cropped image dimensions for visualization
    pca_image_upscaled = cv2.resize(pca_image, (896, 896), interpolation=cv2.INTER_LINEAR)
    output_pca_image = Image.fromarray((pca_image_upscaled * 255).astype(np.uint8))
    output_pca_image.save(f'{fig_name.split(".")[0]}_output.png')

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
    plt.savefig(fig_name)
    plt.show()
    plt.close()

    # Print explained variance ratio (how much variance each PC captures)
    print("\nExplained variance ratio by principal components:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.4f}")
    print(f"  Total explained by 3 PCs: {pca.explained_variance_ratio_.sum():.4f}")


def run_pca_visualization_on_untrained_dino():
    # 1. Load the DINOv2 Model
    # Choose the model size you are using (e.g., 'dinov2_vitl14_reg' for ViT-L with registers)
    # Ensure you use the _reg version if your model was trained with registers,
    # or the non-_reg version if it wasn't.
    # For PCA visualization, you generally use the pre-trained model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'dinov2_vitl14_reg' # Or 'dinov2_vitb14', 'dinov2_vits14', etc.
    dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
    dinov2_model.eval() # Set to evaluation mode

    # DINOv2 models expect specific input image sizes and normalization
    # The global_crops_size is usually 224 for standard models or 518 for larger ones
    # Check your config for the exact global_crops_size and patch_size
    patch_size = 14 # Typically 14 or 16
    global_crop_size = 518 # Or 518, matching your training config

    # clahe_transform = CLAHETransform(
    #     clipLimit=1.0,
    #     tileGridSize=(8,8)
    # )

    normalize = T.Compose(
        [
            T.ToTensor(),
            make_normalize_transform(), # need to edit mean and std
        ]
    )

    transform = T.Compose([
        T.Resize(global_crop_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(global_crop_size), # Or T.RandomResizedCrop if visualizing training data
        #clahe_transform,
        normalize,
    ])

    # 2. Load and Prepare an Image
    for image_path in [image_path_1, image_path_2]:
        if 'fluorescent' in image_path:
            img_type = 'fluorescent'
        else:
            img_type = 'pc_bf'
    
        image = Image.open(image_path).convert('RGB')

        # Original image (for display later)
        original_image_np = np.array(image)

        # Transform the image for DINOv2
        input_image = transform(image).unsqueeze(0).to(device) # Add batch dimension and move to device

        # 3. Extract Patch Features
        with torch.no_grad():
            # 'x_norm_patchtokens' gives normalized patch-level features after the last layer
            # These are typically what's used for PCA visualization
            features_dict = dinov2_model.forward_features(input_image)
            patch_features = features_dict['x_norm_patchtokens'] # Shape: (1, num_patches, feature_dim)
            
            # Remove batch dimension: (num_patches, feature_dim)
            patch_features = patch_features.squeeze(0).cpu().numpy()

        # Determine grid size based on input image size and patch size
        h, w = input_image.shape[-2:] # Cropped image height and width
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        # Verify num_patches matches
        expected_num_patches = num_patches_h * num_patches_w
        if patch_features.shape[0] != expected_num_patches:
            print(f"Warning: Expected {expected_num_patches} patches, got {patch_features.shape[0]}. Check input size and patch size.")
            # This might happen if using reg tokens, they are usually removed before PCA
            # If reg tokens are present, they'll be at the beginning or end.
            # For DINOv2 reg, x_norm_patchtokens should already exclude them.

        # 4. Apply PCA
        # PCA to 3 components (for RGB visualization)
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(patch_features)

        threshold_value = 0.0
        foreground_mask = pca_features[:, 0] > threshold_value
        
        # pca_features[~foreground_mask] = 0 #ADD IF WANT BACKGROUND THRESHOLDING

        # Normalize PCA components to 0-255 for visualization
        # Min-Max scale each component independently
        pca_features_normalized = np.zeros_like(pca_features)
        for i in range(pca_features.shape[1]):
            min_val = pca_features[:, i].min()
            max_val = pca_features[:, i].max()
            if max_val - min_val > 0:
                pca_features_normalized[:, i] = (pca_features[:, i] - min_val) / (max_val - min_val)
            else:
                pca_features_normalized[:, i] = 0.5 # Handle case where component has no variance

        pca_image = pca_features_normalized.reshape(num_patches_h, num_patches_w, 3) # Reshape to a grid

        # 5. Upscale and Visualize
        # Upscale the PCA image to the original cropped image dimensions for visualization
        pca_image_upscaled = cv2.resize(pca_image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Display the results
        os.makedirs(f'../test/pca_visualizations/{img_type}/image_{index}', exist_ok=True)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(original_image_np)
        plt.title("Original Image (Cropped)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pca_image_upscaled)
        plt.title(f"DINOv2 PCA Visualization ({model_name})")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'pca_visualizations/{img_type}/image_{index}/untrained_dino.png')
        plt.close()
        print(f'saving as: pca_visualizations/{img_type}/image_{index}/untrained_dino.png')

        # Print explained variance ratio (how much variance each PC captures)
        print("\nExplained variance ratio by principal components:")
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {ratio:.4f}")
        print(f"  Total explained by 3 PCs: {pca.explained_variance_ratio_.sum():.4f}")

        # visualize_attention(dinov2_model, input_image, run_type='untrained')



def run_pca_visualization_v1(model):
    iteration = iteration.split('_')[-1]
    # 1. Load the DINOv2 Model
    # Choose the model size you are using (e.g., 'dinov2_vitl14_reg' for ViT-L with registers)
    # Ensure you use the _reg version if your model was trained with registers,
    # or the non-_reg version if it wasn't.
    # For PCA visualization, you generally use the pre-trained model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'dinov2_vitl14_reg' # Or 'dinov2_vitb14', 'dinov2_vits14', etc.
    # dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
    dinov2_model = model.teacher.state_dict()
    # dinov2_model.eval() # Set to evaluation mode

    # DINOv2 models expect specific input image sizes and normalization
    # The global_crops_size is usually 224 for standard models or 518 for larger ones
    # Check your config for the exact global_crops_size and patch_size
    patch_size = 14 # Typically 14 or 16
    global_crop_size = 518 # Or 518, matching your training config

    vis_model = DinoVisionTransformer(
        img_size=global_crop_size,
        patch_size=patch_size,
    ).to(device)

    # vis_model = vit_large(
    #     img_size=global_crop_size,
    #     patch_size=patch_size
    # ).to(device)

    # clahe_transform = CLAHETransform(
    #     clipLimit=1.0,
    #     tileGridSize=(8,8)
    # )

    normalize = T.Compose(
        [
            T.ToTensor(),
            make_normalize_transform(), # need to edit mean and std
        ]
    )

    transform = T.Compose([
        T.Resize(global_crop_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(global_crop_size), # Or T.RandomResizedCrop if visualizing training data
        #clahe_transform,
        normalize,
    ])

    # 2. Load and Prepare an 
    for image_path in [image_path_1, image_path_2]:
        if 'fluorescent' in image_path:
            img_type = 'fluorescent'
        else:
            img_type = 'pc_bf'

        image = Image.open(image_path).convert('RGB')

        # Original image (for display later)
        original_image_np = np.array(image)

        # Transform the image for DINOv2
        input_image = transform(image).unsqueeze(0).to(device) # Add batch dimension and move to device

        # 3. Extract Patch Features
        with torch.no_grad():
            # 'x_norm_patchtokens' gives normalized patch-level features after the last layer
            # These are typically what's used for PCA visualization
            features_dict = vis_model.forward_features(input_image)
            patch_features = features_dict['x_norm_patchtokens'] # Shape: (1, num_patches, feature_dim)
            
            # Remove batch dimension: (num_patches, feature_dim)
            patch_features = patch_features.squeeze(0).cpu().numpy()

        # Determine grid size based on input image size and patch size
        h, w = input_image.shape[-2:] # Cropped image height and width
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        # Verify num_patches matches
        expected_num_patches = num_patches_h * num_patches_w
        if patch_features.shape[0] != expected_num_patches:
            print(f"Warning: Expected {expected_num_patches} patches, got {patch_features.shape[0]}. Check input size and patch size.")
            # This might happen if using reg tokens, they are usually removed before PCA
            # If reg tokens are present, they'll be at the beginning or end.
            # For DINOv2 reg, x_norm_patchtokens should already exclude them.

        # 4. Apply PCA
        # PCA to 3 components (for RGB visualization)
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(patch_features)

        threshold_value = 0.0
        foreground_mask = pca_features[:, 0] > threshold_value
        
        # pca_features[~foreground_mask] = 0 #ADD IF WANT BACKGROUND THRESHOLDING

        # Normalize PCA components to 0-255 for visualization
        # Min-Max scale each component independently
        pca_features_normalized = np.zeros_like(pca_features)
        for i in range(pca_features.shape[1]):
            min_val = pca_features[:, i].min()
            max_val = pca_features[:, i].max()
            if max_val - min_val > 0:
                pca_features_normalized[:, i] = (pca_features[:, i] - min_val) / (max_val - min_val)
            else:
                pca_features_normalized[:, i] = 0.5 # Handle case where component has no variance

        pca_image = pca_features_normalized.reshape(num_patches_h, num_patches_w, 3) # Reshape to a grid

        # 5. Upscale and Visualize
        # Upscale the PCA image to the original cropped image dimensions for visualization
        os.makedirs(f'../test/pca_visualizations/{img_type}/image_{index}', exist_ok=True)
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
        plt.savefig(f'../test/pca_visualizations/{img_type}/image_{index}/random_dino.png')
        plt.close()
        print(f'saving as: ../test/pca_visualizations/{img_type}/image_{index}/random_dino.png')

        # Print explained variance ratio (how much variance each PC captures)
        print("\nExplained variance ratio by principal components:")
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {ratio:.4f}")
        print(f"  Total explained by 3 PCs: {pca.explained_variance_ratio_.sum():.4f}")






if __name__ == '__main__':
    run_pca_visualization_on_untrained_dino()

