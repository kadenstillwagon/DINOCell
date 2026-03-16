# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note the original is here: https://github.com/facebookresearch/dino/blob/main/visualize_attention.py

import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from dinov2.models.vision_transformer import vit_small, vit_large
from dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.models import build_model_from_cfg

from dinov2.data.transforms import CLAHETransform, make_normalize_transform
from torchvision import transforms as T



index = 30000
image_path_1 = f'../../../SSL_Training_Dataset_1024/train/pc_bf/{os.listdir(f"../../../SSL_Training_Dataset_1024/train/pc_bf")[index]}'
image_path_2 = f'../../../SSL_Training_Dataset_1024/train/fluorescent/{os.listdir(f"../../../SSL_Training_Dataset_1024/train/fluorescent")[index]}'


def run_attention_visualization(model, cfg, model_version, iteration):
    iteration = iteration.split('_')[-1]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    patch_size = 14 # Typically 14 or 16
    global_crop_size = 518 # Or 518, matching your training config

    _, teacher_backbone, _ = build_model_from_cfg(cfg)
    teacher_backbone.to(device)
    teacher_backbone.load_state_dict(model.teacher.backbone.state_dict(), strict=False)


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

    for image_path in [image_path_1, image_path_2]:
        if 'fluorescent' in image_path:
            img_type = 'fluorescent'
        else:
            img_type = 'pc_bf'
    
        img = Image.open(image_path).convert('RGB')

        # Original image (for display later)
        original_image_np = np.array(img)

        img = transform(img)

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        with torch.no_grad():
            attentions = teacher_backbone.get_last_self_attention(img.to(device))
            # print(attentions.shape)

        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        # for every patch
        attentions = attentions[0, :, 0, 5:].reshape(nh, -1)
        # weird: one pixel gets high attention over all heads?
        # print(torch.max(attentions, dim=1)) 
        # attentions[:, 283] = 0 

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="bilinear")[0].cpu().numpy()

        # save attentions heatmaps
        os.makedirs(f'../test/attention_visualizations/{img_type}/image_{index}', exist_ok=True)
        os.makedirs(f'../test/attention_visualizations/{img_type}/image_{index}/finetuned_dino_{model_version}_iteration_{iteration}', exist_ok=True)
        
        for j in range(nh):
            fname = f'../test/attention_visualizations/{img_type}/image_{index}/finetuned_dino_{model_version}_iteration_{iteration}/head_{j}.png'
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(original_image_np)
            plt.title("Original Image (Cropped)")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(attentions[j])
            plt.title(f"DINOv2 Attention Visualization")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"{fname} saved.")




def run_attention_visualization_untrained_dino():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    patch_size = 14 # Typically 14 or 16
    global_crop_size = 518 # Or 518, matching your training config

    model = vit_large(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            #ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=4
    )

    model.load_state_dict(torch.load('../pretrained_models/dinov2_vitl14_reg4_pretrain.pth'))
    # model.load_state_dict(torch.load('../outputs/model_0002799.rank_0.pth', weights_only=False))
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()


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

    for image_path in [image_path_1, image_path_2]:
        if 'fluorescent' in image_path:
            img_type = 'fluorescent'
        else:
            img_type = 'pc_bf'
    
        img = Image.open(image_path).convert('RGB')

        # Original image (for display later)
        original_image_np = np.array(image)

        img = transform(img)
        print(img.shape)

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        attentions = model.get_last_self_attention(img.to(device))
        print(attentions.shape)

        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        # for every patch
        attentions = attentions[0, :, 0, 5:].reshape(nh, -1)
        # weird: one pixel gets high attention over all heads?
        print(torch.max(attentions, dim=1)) 
        # attentions[:, 283] = 0 

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="bilinear")[0].cpu().numpy()

        # save attentions heatmaps
        os.makedirs(f'../test/attention_visualizations/{img_type}/image_{index}', exist_ok=True)
        os.makedirs(f'../test/attention_visualizations/{img_type}/image_{index}/untrained_dino', exist_ok=True)
        
        for j in range(nh):
            fname = f'../test/attention_visualizations/{img_type}/image_{index}/untrained_dino/head_{j}.png'
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(original_image_np)
            plt.title("Original Image (Cropped)")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(attentions[j])
            plt.title(f"DINOv2 Attention Visualization")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"{fname} saved.")


if __name__ == '__main__':
    run_attention_visualization_untrained_dino()