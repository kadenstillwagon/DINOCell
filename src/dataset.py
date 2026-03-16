from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import cv2
from torchvision import transforms
import torch
import random
import time
import sys
from PIL import Image
from utils import datasets_that_need_ignore_masking, silver_dataset, ood_datasets, RandomScale, RandomInvertContrast, PoissonNoise, RandomDownsampleUpsample, RandomSpeckleNoise


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dinov2.dinov2.data.transforms import make_normalize_transform

import matplotlib.pyplot as plt


def get_dataset(objective_type, dataset_root_path, dataset_list, split, include_silver_data, use_advanced_augmentations, ignore_all, ignore_mask_weight, silver_data_weight, crop_size=256):
    if objective_type == 'distance_map':
        return DINODistanceMapDataset(dataset_root_path, dataset_list, split, include_silver_data, use_advanced_augmentations, ignore_all, ignore_mask_weight, silver_data_weight, crop_size)
    elif objective_type == 'directional_distance_map':
        return DINODirectionalDistanceMapDataset(dataset_root_path, dataset_list, split, include_silver_data, use_advanced_augmentations, ignore_all, ignore_mask_weight, silver_data_weight, crop_size)
    elif objective_type == 'flows':
        return DINOFlowsDataset(dataset_root_path, dataset_list, split, include_silver_data, use_advanced_augmentations, ignore_all, ignore_mask_weight, silver_data_weight, crop_size)
    elif objective_type == 'direct_mask_prediction' or objective_type == 'direct_mask_prediction_independent_encoding':
        return DINODirectMaskDataset(dataset_root_path, dataset_list, split, include_silver_data, use_advanced_augmentations, ignore_all, ignore_mask_weight, silver_data_weight, crop_size)


class DINOCellOODDataset(Dataset):
    def __init__(self, dataset_path, use_advanced_augmentations, crop_size):
        self.use_advanced_augmentations = use_advanced_augmentations
        self.img_files = [f'{dataset_path}/imgs/{f}' for f in np.sort(os.listdir(f'{dataset_path}/imgs')) if f.endswith('.png')]
        self.ann_files = [f'{dataset_path}/anns/{f}' for f in np.sort(os.listdir(f'{dataset_path}/anns')) if f.endswith('.npy')]
        if os.path.exists(f'{dataset_path}/ignore_masks'):
            self.ignore_mask_files = [f'{dataset_path}/ignore_masks/{f}' for f in np.sort(os.listdir(f'{dataset_path}/ignore_masks')) if f.endswith('.npy')]
        else:
            self.ignore_mask_files = [None for f in os.listdir(f'{dataset_path}/anns') if f.endswith('.npy')]
            

        self.crop_size = crop_size

    def __len__(self):
        return len(self.img_files)

    def _resize(self, img, ann, ignore_mask):
        
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        #convert to image grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #resize if needed:
        if img.shape[0] < self.crop_size and img.shape[1] < self.crop_size:
            if img.shape[0] < img.shape[1]:
                multiplier = self.crop_size / img.shape[0]
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
                ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
            else:
                multiplier = self.crop_size / img.shape[1]
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
                ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
        elif img.shape[0] < self.crop_size:
            multiplier = self.crop_size / img.shape[0]
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
            ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
            ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
        elif img.shape[1] < self.crop_size:
            multiplier = self.crop_size / img.shape[1]
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
            ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
            ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img, ann, ignore_mask

    def _preprocess(self, img, use_advanced_augmentations=False):
        if use_advanced_augmentations:
            #Percentile Normalization
            p1, p99 = np.percentile(img, (1, 99))
            img = np.clip((img - p1) / (p99 - p1), 0, 1)

            #Z-Score Normalization
            # img = (img - img.mean()) / (img.std() + 1e-8)

            if isinstance(img, np.ndarray):
                img = img.astype(np.float32)
                #cvt to 3 channel
                if len(img.shape) < 3 or img.shape[2] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = img.to(torch.float32)
                img = img.permute(1, 2, 0)
                img = img.cpu().numpy()
                #cvt to 3 channel
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

            #Not Doing CLAHE
        else:
            #adaptive hist normalization
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)
            img = cv2.normalize(img_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            #cvt to 3 channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_files[idx]))
        ann = np.load(self.ann_files[idx])
        if self.ignore_mask_files[idx] is not None:
            ignore_mask = np.load(self.ignore_mask_files[idx])
        else:
            ignore_mask = np.zeros_like(ann)

        #preprocess image
        img = self._preprocess(img, self.use_advanced_augmentations)
        
        #resize everything
        img, ann, ignore_mask = self._resize(img, ann, ignore_mask)

        #random crop to size
        x = random.randint(0, img.shape[1] - self.crop_size)
        y = random.randint(0, img.shape[0] - self.crop_size)
        img = img[y:y+self.crop_size, x:x+self.crop_size]
        ann = ann[y:y+self.crop_size, x:x+self.crop_size]
        ignore_mask = ignore_mask[y:y+self.crop_size, x:x+self.crop_size]
        
        #Invert Ignore Mask
        ignore_mask = np.where(ignore_mask > 0, 0, 1)
        
        # prepare image and prompt for the model
        # inputs = self.processor(img, return_tensors="pt") #input image: shape: (256, 256, 3), range [0, 255]
        if self.use_advanced_augmentations:
            if self.crop_size != 896:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # make_normalize_transform(),
                        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
                    ]
                )
            else:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # make_normalize_transform()
                    ]
                )
        else:
            if self.crop_size != 896:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform(),
                        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
                    ]
                )
            else:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform()
                    ]
                )
        img = image_transform(img)
        inputs = {'pixel_values': img}

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        gt_mask_inputs = {}
        gt_mask_inputs['gt_mask'] = ann
        gt_mask_inputs['ignore_mask'] = ignore_mask

        inputs['gt_data'] = gt_mask_inputs
        # inputs['binary_mask'] = label > 0.001

        return inputs
    
###################################
#  TRAINING AND TESTING DATASETS
###################################

class DINODistanceMapDataset(Dataset):
    def __init__(self, dataset_root_path, dataset_list, split, include_silver_data, use_advanced_augmentations, ignore_all, ignore_mask_weight, silver_data_weight, crop_size):
        
        self.include_silver_data = include_silver_data
        self.use_advanced_augmentations = use_advanced_augmentations
        self.ignore_all = ignore_all
        self.ignore_mask_weight = ignore_mask_weight
        self.silver_data_weight = silver_data_weight
        self.crop_size = crop_size
        self.isTrain = 'train' == split

        self.img_files = []
        self.ann_files = []
        self.dist_map_files = []
        self.ignore_mask_files = []
        self.example_weights = []

        for dataset in dataset_list:
            if dataset not in ood_datasets:
                # dataset_path = f'{dataset_root_path}/{dataset}/{split}'

                # self.img_files.extend([f'{dataset_path}/imgs/{f}' for f in np.sort(os.listdir(f'{dataset_path}/imgs')) if f.endswith('.png')])
                # self.ann_files.extend([f'{dataset_path}/anns/{f}' for f in np.sort(os.listdir(f'{dataset_path}/anns')) if f.endswith('.npy')])
                # self.dist_map_files.extend([f'{dataset_path}/dist_maps/{f}' for f in np.sort(os.listdir(f'{dataset_path}/dist_maps')) if f.endswith('.npy')])
                # if not ignore_all and dataset in datasets_that_need_ignore_masking:
                #     self.ignore_mask_files.extend([None for f in os.listdir(f'{dataset_path}/ignore_masks') if f.endswith('.npy')])
                # else:
                #     self.ignore_mask_files.extend([f'{dataset_path}/ignore_masks/{f}' for f in np.sort(os.listdir(f'{dataset_path}/ignore_masks')) if f.endswith('.npy')])

                # if dataset in silver_dataset:
                #     self.example_weights.extend([silver_data_weight for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])
                # else:
                #     self.example_weights.extend([1.0 for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])

                # USE ONCE 512 TRAINING DONE
                if not self.isTrain and dataset in silver_dataset:
                    continue
                else:
                    if not self.include_silver_data and dataset in silver_dataset:
                        continue
                    else:
                        dataset_path = f'{dataset_root_path}/{dataset}/{split}'

                        self.img_files.extend([f'{dataset_path}/imgs/{f}' for f in np.sort(os.listdir(f'{dataset_path}/imgs')) if f.endswith('.png')])
                        self.ann_files.extend([f'{dataset_path}/anns/{f}' for f in np.sort(os.listdir(f'{dataset_path}/anns')) if f.endswith('.npy')])
                        self.dist_map_files.extend([f'{dataset_path}/dist_maps/{f}' for f in np.sort(os.listdir(f'{dataset_path}/dist_maps')) if f.endswith('.npy')])
                        if not ignore_all and dataset not in datasets_that_need_ignore_masking:
                            self.ignore_mask_files.extend([None for f in os.listdir(f'{dataset_path}/ignore_masks') if f.endswith('.npy')])
                        else:
                            self.ignore_mask_files.extend([f'{dataset_path}/ignore_masks/{f}' for f in np.sort(os.listdir(f'{dataset_path}/ignore_masks')) if f.endswith('.npy')])

                        if dataset in silver_dataset:
                            self.example_weights.extend([silver_data_weight for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])
                        else:
                            self.example_weights.extend([1.0 for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])
                



    def __len__(self):
        return len(self.img_files)

    def _resize(self, img, ann, ignore_mask, dist_map):
        #convert to image grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #resize if needed:
        if img.shape[0] < self.crop_size and img.shape[1] < self.crop_size:
            if img.shape[0] < img.shape[1]:
                multiplier = self.crop_size / img.shape[0]
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                dist_map = cv2.resize(dist_map, (int(multiplier * dist_map.shape[1]) + 1, int(multiplier * dist_map.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            else:
                multiplier = self.crop_size / img.shape[1]
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                dist_map = cv2.resize(dist_map, (int(multiplier * dist_map.shape[1]) + 1, int(multiplier * dist_map.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
        elif img.shape[0] < self.crop_size:
            multiplier = self.crop_size / img.shape[0]
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            dist_map = cv2.resize(dist_map, (int(multiplier * dist_map.shape[1]) + 1, int(multiplier * dist_map.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
        elif img.shape[1] < self.crop_size:
            multiplier = self.crop_size / img.shape[1]
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            dist_map = cv2.resize(dist_map, (int(multiplier * dist_map.shape[1]) + 1, int(multiplier * dist_map.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img, ann, ignore_mask, dist_map

    def _preprocess(self, img, use_advanced_augmentations=False):
        if use_advanced_augmentations:
            #Percentile Normalization
            p1, p99 = np.percentile(img, (1, 99))
            img = np.clip((img - p1) / (p99 - p1), 0, 1)

            #Z-Score Normalization
            # img = (img - img.mean()) / (img.std() + 1e-8)

            if isinstance(img, np.ndarray):
                img = img.astype(np.float32)
                #cvt to 3 channel
                if len(img.shape) < 3 or img.shape[2] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = img.to(torch.float32)
                img = img.permute(1, 2, 0)
                img = img.cpu().numpy()
                #cvt to 3 channel
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

            #Not Doing CLAHE
        else:
            #adaptive hist normalization
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)
            img = cv2.normalize(img_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            #cvt to 3 channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img
    
    def _data_augmentation(self, img, ignore_mask, dist_map, use_advanced_augmentations=False):
        if use_advanced_augmentations:
            #prevent overflow
            img = img.to(torch.float32).permute(2, 0, 1)
            ignore_mask = torch.Tensor(ignore_mask).to(torch.float32).unsqueeze(0)
            dist_map = torch.Tensor(dist_map).to(torch.float32).unsqueeze(0)

            #geometric augmentations - apply to image and labels
            random_scale_aug = RandomScale(p=0.25, scale_range=(0.25, 4.0))
            rescaled_data = random_scale_aug([img, ignore_mask, dist_map])

            img = rescaled_data[0]
            ignore_mask = rescaled_data[1].squeeze(0).cpu().numpy()
            ignore_mask = np.where(ignore_mask > 0.5, 1, 0)
            dist_map = rescaled_data[2].squeeze(0).cpu().numpy()

            intensity_augmentations = transforms.Compose(
                [
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
                    RandomInvertContrast(p=0.25)
                ]
            )

            rand = random.random()
            if rand <= 0.125:
                intensity_augmentations.transforms.append(PoissonNoise(lam_scale=30))
            elif rand > 0.125 and rand <= 0.25:
                intensity_augmentations.transforms.append(transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)))
            elif rand > 0.25 and rand <= 0.375:
                intensity_augmentations.transforms.append(RandomDownsampleUpsample(scale_range=(0.5, 0.9)))
            elif rand > 0.375 and rand <= 0.5:
                intensity_augmentations.transforms.append(RandomSpeckleNoise(std=0.05))

            img = torch.clamp((img - img.min()) / (img.max() - img.min()), 0.0, 1.0)
            img = intensity_augmentations(img)
            img = self._preprocess(img, use_advanced_augmentations)
            img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
        else:
            #prevent overflow
            img = img.astype(np.float32)

            #random brightness
            alpha = random.uniform(0.95, 1.05)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

            # #random contrast / gamma
            # beta = random.randint(-1, 1)
            # gamma = random.uniform(0.9, 1.1)
            # img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
            # img = np.power(img, gamma)

            #back to uint8
            img = np.clip(img, 0, 255).astype(np.uint8)

            #randomly invert image
            if random.random() > 0.5:
                img = cv2.bitwise_not(img)

        return img, ignore_mask, dist_map

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_files[idx]))
        ann = np.load(self.ann_files[idx])
        dist_map = np.load(self.dist_map_files[idx])
        example_weight = self.example_weights[idx]
        if self.ignore_mask_files[idx] is not None:
            ignore_mask = np.load(self.ignore_mask_files[idx])
        else:
            ignore_mask = np.zeros_like(ann)

        #preprocess image
        img = self._preprocess(img, self.use_advanced_augmentations)

        if self.isTrain:
            #data augmentation
            img, ignore_mask, dist_map = self._data_augmentation(img, ignore_mask, dist_map, self.use_advanced_augmentations)
            ann = np.zeros_like(img)
        
        #resize everything
        img, ann, ignore_mask, dist_map = self._resize(img, ann, ignore_mask, dist_map)

        #random crop to size
        x = random.randint(0, img.shape[1] - self.crop_size)
        y = random.randint(0, img.shape[0] - self.crop_size)
        img = img[y:y+self.crop_size, x:x+self.crop_size]
        ann = ann[y:y+self.crop_size, x:x+self.crop_size]
        ignore_mask = ignore_mask[y:y+self.crop_size, x:x+self.crop_size]
        dist_map = dist_map[y:y+self.crop_size, x:x+self.crop_size]
        
        #Invert Ignore Mask
        ignore_mask = np.where(ignore_mask > 0, self.ignore_mask_weight, 1)
        
        # prepare image and prompt for the model
        # inputs = self.processor(img, return_tensors="pt") #input image: shape: (256, 256, 3), range [0, 255]
        if self.use_advanced_augmentations:
            if self.crop_size != 896:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # make_normalize_transform(),
                        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
                    ]
                )
            else:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # make_normalize_transform()
                    ]
                )
        else:
            if self.crop_size != 896:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform(),
                        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
                    ]
                )
            else:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform()
                    ]
                )
        img = image_transform(img)
        inputs = {'pixel_values': img}

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        gt_mask_inputs = {}
        gt_mask_inputs['gt_mask'] = ann
        gt_mask_inputs['ignore_mask'] = ignore_mask
        gt_mask_inputs['gt_dist_map'] = dist_map
        gt_mask_inputs['example_weight'] = example_weight

        inputs['gt_data'] = gt_mask_inputs
        # inputs['binary_mask'] = label > 0.001

        return inputs




class DINODirectionalDistanceMapDataset(Dataset):
    def __init__(self, dataset_root_path, dataset_list, split, include_silver_data, use_advanced_augmentations, ignore_all, ignore_mask_weight, silver_data_weight, crop_size):
        
        self.include_silver_data = include_silver_data
        self.use_advanced_augmentations = use_advanced_augmentations
        self.ignore_all = ignore_all
        self.ignore_mask_weight = ignore_mask_weight
        self.silver_data_weight = silver_data_weight
        self.crop_size = crop_size
        self.isTrain = 'train' == split

        self.img_files = []
        self.ann_files = []
        self.dist_map_files = []
        self.directional_dist_map_files = []
        self.ignore_mask_files = []
        self.example_weights = []

        for dataset in dataset_list:
            if dataset not in ood_datasets:
                # dataset_path = f'{dataset_root_path}/{dataset}/{split}'

                # self.img_files.extend([f'{dataset_path}/imgs/{f}' for f in np.sort(os.listdir(f'{dataset_path}/imgs')) if f.endswith('.png')])
                # self.ann_files.extend([f'{dataset_path}/anns/{f}' for f in np.sort(os.listdir(f'{dataset_path}/anns')) if f.endswith('.npy')])
                # self.dist_map_files.extend([f'{dataset_path}/dist_maps/{f}' for f in np.sort(os.listdir(f'{dataset_path}/dist_maps')) if f.endswith('.npy')])
                # self.directional_dist_map_files.extend([f'{dataset_path}/4d_dist_maps_norm/{f}' for f in np.sort(os.listdir(f'{dataset_path}/4d_dist_maps_norm')) if f.endswith('.npy')])
                # if not ignore_all and dataset in datasets_that_need_ignore_masking:
                #     self.ignore_mask_files.extend([None for f in os.listdir(f'{dataset_path}/ignore_masks') if f.endswith('.npy')])
                # else:
                #     self.ignore_mask_files.extend([f'{dataset_path}/ignore_masks/{f}' for f in np.sort(os.listdir(f'{dataset_path}/ignore_masks')) if f.endswith('.npy')])

                # if dataset in silver_dataset:
                #     self.example_weights.extend([silver_data_weight for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])
                # else:
                #     self.example_weights.extend([1.0 for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])

                # USE ONCE 512 TRAINING DONE
                if not self.isTrain and dataset in silver_dataset:
                    continue
                else:
                    if not self.include_silver_data and dataset in silver_dataset:
                        continue
                    else:
                        dataset_path = f'{dataset_root_path}/{dataset}/{split}'

                        self.img_files.extend([f'{dataset_path}/imgs/{f}' for f in np.sort(os.listdir(f'{dataset_path}/imgs')) if f.endswith('.png')])
                        self.ann_files.extend([f'{dataset_path}/anns/{f}' for f in np.sort(os.listdir(f'{dataset_path}/anns')) if f.endswith('.npy')])
                        self.dist_map_files.extend([f'{dataset_path}/dist_maps/{f}' for f in np.sort(os.listdir(f'{dataset_path}/dist_maps')) if f.endswith('.npy')])
                        self.directional_dist_map_files.extend([f'{dataset_path}/4d_dist_maps_norm/{f}' for f in np.sort(os.listdir(f'{dataset_path}/4d_dist_maps_norm')) if f.endswith('.npy')])
                        if not ignore_all and dataset not in datasets_that_need_ignore_masking:
                            self.ignore_mask_files.extend([None for f in os.listdir(f'{dataset_path}/ignore_masks') if f.endswith('.npy')])
                        else:
                            self.ignore_mask_files.extend([f'{dataset_path}/ignore_masks/{f}' for f in np.sort(os.listdir(f'{dataset_path}/ignore_masks')) if f.endswith('.npy')])

                        if dataset in silver_dataset:
                            self.example_weights.extend([silver_data_weight for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])
                        else:
                            self.example_weights.extend([1.0 for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])

        


    def __len__(self):
        return len(self.img_files)

    def _resize(self, img, ann, ignore_mask, dist_map, dist_map_top, dist_map_bottom, dist_map_left, dist_map_right):
        #convert to image grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #resize if needed:
        if img.shape[0] < self.crop_size and img.shape[1] < self.crop_size:
            if img.shape[0] < img.shape[1]:
                multiplier = self.crop_size / img.shape[0]
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                dist_map = cv2.resize(dist_map, (int(multiplier * dist_map.shape[1]) + 1, int(multiplier * dist_map.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                dist_map_top = cv2.resize(dist_map_top, (int(multiplier * dist_map_top.shape[1]) + 1, int(multiplier * dist_map_top.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                dist_map_bottom = cv2.resize(dist_map_bottom, (int(multiplier * dist_map_bottom.shape[1]) + 1, int(multiplier * dist_map_bottom.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                dist_map_left = cv2.resize(dist_map_left, (int(multiplier * dist_map_left.shape[1]) + 1, int(multiplier * dist_map_left.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                dist_map_right = cv2.resize(dist_map_right, (int(multiplier * dist_map_right.shape[1]) + 1, int(multiplier * dist_map_right.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            else:
                multiplier = self.crop_size / img.shape[1]
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                dist_map = cv2.resize(dist_map, (int(multiplier * dist_map.shape[1]) + 1, int(multiplier * dist_map.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                dist_map_top = cv2.resize(dist_map_top, (int(multiplier * dist_map_top.shape[1]) + 1, int(multiplier * dist_map_top.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                dist_map_bottom = cv2.resize(dist_map_bottom, (int(multiplier * dist_map_bottom.shape[1]) + 1, int(multiplier * dist_map_bottom.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                dist_map_left = cv2.resize(dist_map_left, (int(multiplier * dist_map_left.shape[1]) + 1, int(multiplier * dist_map_left.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                dist_map_right = cv2.resize(dist_map_right, (int(multiplier * dist_map_right.shape[1]) + 1, int(multiplier * dist_map_right.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
        elif img.shape[0] < self.crop_size:
            multiplier = self.crop_size / img.shape[0]
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            dist_map = cv2.resize(dist_map, (int(multiplier * dist_map.shape[1]) + 1, int(multiplier * dist_map.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            dist_map_top = cv2.resize(dist_map_top, (int(multiplier * dist_map_top.shape[1]) + 1, int(multiplier * dist_map_top.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            dist_map_bottom = cv2.resize(dist_map_bottom, (int(multiplier * dist_map_bottom.shape[1]) + 1, int(multiplier * dist_map_bottom.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            dist_map_left = cv2.resize(dist_map_left, (int(multiplier * dist_map_left.shape[1]) + 1, int(multiplier * dist_map_left.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            dist_map_right = cv2.resize(dist_map_right, (int(multiplier * dist_map_right.shape[1]) + 1, int(multiplier * dist_map_right.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
        elif img.shape[1] < self.crop_size:
            multiplier = self.crop_size / img.shape[1]
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            dist_map = cv2.resize(dist_map, (int(multiplier * dist_map.shape[1]) + 1, int(multiplier * dist_map.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            dist_map_top = cv2.resize(dist_map_top, (int(multiplier * dist_map_top.shape[1]) + 1, int(multiplier * dist_map_top.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            dist_map_bottom = cv2.resize(dist_map_bottom, (int(multiplier * dist_map_bottom.shape[1]) + 1, int(multiplier * dist_map_bottom.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            dist_map_left = cv2.resize(dist_map_left, (int(multiplier * dist_map_left.shape[1]) + 1, int(multiplier * dist_map_left.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            dist_map_right = cv2.resize(dist_map_right, (int(multiplier * dist_map_right.shape[1]) + 1, int(multiplier * dist_map_right.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img, ann, ignore_mask, dist_map, dist_map_top, dist_map_bottom, dist_map_left, dist_map_right

    def _preprocess(self, img, use_advanced_augmentations=False):
        if use_advanced_augmentations:
            #Percentile Normalization
            p1, p99 = np.percentile(img, (1, 99))
            img = np.clip((img - p1) / (p99 - p1), 0, 1)

            #Z-Score Normalization
            # img = (img - img.mean()) / (img.std() + 1e-8)

            if isinstance(img, np.ndarray):
                img = img.astype(np.float32)
                #cvt to 3 channel
                if len(img.shape) < 3 or img.shape[2] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = img.to(torch.float32)
                img = img.permute(1, 2, 0)
                img = img.cpu().numpy()
                #cvt to 3 channel
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

            #Not Doing CLAHE
        else:
            #adaptive hist normalization
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)
            img = cv2.normalize(img_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            #cvt to 3 channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img
    
    def _data_augmentation(self, img, ignore_mask, dist_map, dist_map_top, dist_map_bottom, dist_map_left, dist_map_right, use_advanced_augmentations=False):
        if use_advanced_augmentations:
            #prevent overflow
            img = torch.Tensor(img).to(torch.float32).permute(2, 0, 1)

            ignore_mask = torch.Tensor(ignore_mask).to(torch.float32).unsqueeze(0)
            dist_map = torch.Tensor(dist_map).to(torch.float32).unsqueeze(0)
            dist_map_top = torch.Tensor(dist_map_top).to(torch.float32).unsqueeze(0)
            dist_map_bottom = torch.Tensor(dist_map_bottom).to(torch.float32).unsqueeze(0)
            dist_map_left = torch.Tensor(dist_map_left).to(torch.float32).unsqueeze(0)
            dist_map_right = torch.Tensor(dist_map_right).to(torch.float32).unsqueeze(0)

            #geometric augmentations - apply to image and labels
            random_scale_aug = RandomScale(p=0.25, scale_range=(0.25, 4.0))
            rescaled_data = random_scale_aug([img, ignore_mask, dist_map, dist_map_top, dist_map_bottom, dist_map_left, dist_map_right])


            img = rescaled_data[0]
            ignore_mask = rescaled_data[1].squeeze(0).cpu().numpy()
            ignore_mask = np.where(ignore_mask > 0.5, 1, 0)
            dist_map = rescaled_data[2].squeeze(0).cpu().numpy()
            dist_map_top = rescaled_data[3].squeeze(0).cpu().numpy()
            dist_map_bottom = rescaled_data[4].squeeze(0).cpu().numpy()
            dist_map_left = rescaled_data[5].squeeze(0).cpu().numpy()
            dist_map_right = rescaled_data[6].squeeze(0).cpu().numpy()

            intensity_augmentations = transforms.Compose(
                [
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
                    RandomInvertContrast(p=0.25)
                ]
            )

            rand = random.random()
            if rand <= 0.125:
                intensity_augmentations.transforms.append(PoissonNoise(lam_scale=30))
            elif rand > 0.125 and rand <= 0.25:
                intensity_augmentations.transforms.append(transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)))
            elif rand > 0.25 and rand <= 0.375:
                intensity_augmentations.transforms.append(RandomDownsampleUpsample(scale_range=(0.5, 0.9)))
            elif rand > 0.375 and rand <= 0.5:
                intensity_augmentations.transforms.append(RandomSpeckleNoise(std=0.05))

            
            img = torch.clamp((img - img.min()) / (img.max() - img.min()), 0.0, 1.0)
            img = intensity_augmentations(img)
            img = self._preprocess(img, use_advanced_augmentations)
            img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
        else:
            #prevent overflow
            img = img.astype(np.float32)

            #random brightness
            alpha = random.uniform(0.95, 1.05)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

            # #random contrast / gamma
            # beta = random.randint(-1, 1)
            # gamma = random.uniform(0.9, 1.1)
            # img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
            # img = np.power(img, gamma)

            #back to uint8
            img = np.clip(img, 0, 255).astype(np.uint8)

            #randomly invert image
            if random.random() > 0.5:
                img = cv2.bitwise_not(img)

        return img, ignore_mask, dist_map, dist_map_top, dist_map_bottom, dist_map_left, dist_map_right

    def __getitem__(self, idx):
        # print(self.img_files[idx])
        img = np.array(Image.open(self.img_files[idx]))
        ann = np.load(self.ann_files[idx])
        dist_map = np.load(self.dist_map_files[idx])
        directional_dist_map = np.load(self.directional_dist_map_files[idx])
        dist_map_top, dist_map_bottom, dist_map_left, dist_map_right = directional_dist_map[0], directional_dist_map[1], directional_dist_map[2], directional_dist_map[3]
        example_weight = self.example_weights[idx]
        if self.ignore_mask_files[idx] is not None:
            ignore_mask = np.load(self.ignore_mask_files[idx])
        else:
            ignore_mask = np.zeros_like(ann)


        #preprocess image
        img = self._preprocess(img, self.use_advanced_augmentations)

        if self.isTrain:
            #data augmentation
            img, ignore_mask, dist_map, dist_map_top, dist_map_bottom, dist_map_left, dist_map_right = self._data_augmentation(img, ignore_mask, dist_map, dist_map_top, dist_map_bottom, dist_map_left, dist_map_right, self.use_advanced_augmentations)
            ann = np.zeros_like(img)

        #resize everything
        img, ann, ignore_mask, dist_map, dist_map_top, dist_map_bottom, dist_map_left, dist_map_right = self._resize(img, ann, ignore_mask, dist_map, dist_map_top, dist_map_bottom, dist_map_left, dist_map_right)

        #random crop to size
        x = random.randint(0, img.shape[1] - self.crop_size)
        y = random.randint(0, img.shape[0] - self.crop_size)
        img = img[y:y+self.crop_size, x:x+self.crop_size]
        ann = ann[y:y+self.crop_size, x:x+self.crop_size]
        ignore_mask = ignore_mask[y:y+self.crop_size, x:x+self.crop_size]
        dist_map = dist_map[y:y+self.crop_size, x:x+self.crop_size]
        dist_map_top = dist_map_top[y:y+self.crop_size, x:x+self.crop_size]
        dist_map_bottom = dist_map_bottom[y:y+self.crop_size, x:x+self.crop_size]
        dist_map_left = dist_map_left[y:y+self.crop_size, x:x+self.crop_size]
        dist_map_right = dist_map_right[y:y+self.crop_size, x:x+self.crop_size]

        #Invert Ignore Mask
        ignore_mask = np.where(ignore_mask > 0, self.ignore_mask_weight, 1)


        # prepare image and prompt for the model
        # inputs = self.processor(img, return_tensors="pt") #input image: shape: (256, 256, 3), range [0, 255]
        if self.use_advanced_augmentations:
            if self.crop_size != 896:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # make_normalize_transform(),
                        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
                    ]
                )
            else:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # make_normalize_transform()
                    ]
                )
        else:
            if self.crop_size != 896:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform(),
                        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
                    ]
                )
            else:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform()
                    ]
                )
        img = image_transform(img)
        inputs = {'pixel_values': img}

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        gt_mask_inputs = {}
        gt_mask_inputs['gt_mask'] = ann
        gt_mask_inputs['ignore_mask'] = ignore_mask
        gt_mask_inputs['gt_dist_map'] = dist_map
        gt_mask_inputs['gt_dist_map_top'] = dist_map_top
        gt_mask_inputs['gt_dist_map_bottom'] = dist_map_bottom
        gt_mask_inputs['gt_dist_map_left'] = dist_map_left
        gt_mask_inputs['gt_dist_map_right'] = dist_map_right
        gt_mask_inputs['example_weight'] = example_weight

        inputs['gt_data'] = gt_mask_inputs
        # inputs['binary_mask'] = label > 0.001

        return inputs


class DINOFlowsDataset(Dataset):
    def __init__(self, dataset_root_path, dataset_list, split, include_silver_data, use_advanced_augmentations, ignore_all, ignore_mask_weight, silver_data_weight, crop_size):
        
        self.include_silver_data = include_silver_data
        self.use_advanced_augmentations = use_advanced_augmentations
        self.ignore_all = ignore_all
        self.ignore_mask_weight = ignore_mask_weight
        self.silver_data_weight = silver_data_weight
        self.crop_size = crop_size
        self.isTrain = 'train' == split

        self.img_files = []
        self.ann_files = []
        self.flow_files = []
        self.ignore_mask_files = []
        self.example_weights = []

        for dataset in dataset_list:
            if dataset not in ood_datasets:
                # dataset_path = f'{dataset_root_path}/{dataset}/{split}'

                # self.img_files.extend([f'{dataset_path}/imgs/{f}' for f in np.sort(os.listdir(f'{dataset_path}/imgs')) if f.endswith('.png')])
                # self.ann_files.extend([f'{dataset_path}/anns/{f}' for f in np.sort(os.listdir(f'{dataset_path}/anns')) if f.endswith('.npy')])
                # self.flow_files.extend([f'{dataset_path}/flows/{f}' for f in np.sort(os.listdir(f'{dataset_path}/flows')) if f.endswith('.npy')])
                # if not ignore_all and dataset in datasets_that_need_ignore_masking:
                #     self.ignore_mask_files.extend([None for f in os.listdir(f'{dataset_path}/ignore_masks') if f.endswith('.npy')])
                # else:
                #     self.ignore_mask_files.extend([f'{dataset_path}/ignore_masks/{f}' for f in np.sort(os.listdir(f'{dataset_path}/ignore_masks')) if f.endswith('.npy')])

                # if dataset in silver_dataset:
                #     self.example_weights.extend([silver_data_weight for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])
                # else:
                #     self.example_weights.extend([1.0 for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])

                # USE ONCE 512 TRAINING DONE
                if not self.isTrain and dataset in silver_dataset:
                    continue
                else:
                    if not self.include_silver_data and dataset in silver_dataset:
                        continue
                    else:
                        dataset_path = f'{dataset_root_path}/{dataset}/{split}'

                        self.img_files.extend([f'{dataset_path}/imgs/{f}' for f in np.sort(os.listdir(f'{dataset_path}/imgs')) if f.endswith('.png')])
                        self.ann_files.extend([f'{dataset_path}/anns/{f}' for f in np.sort(os.listdir(f'{dataset_path}/anns')) if f.endswith('.npy')])
                        self.flow_files.extend([f'{dataset_path}/flows/{f}' for f in np.sort(os.listdir(f'{dataset_path}/flows')) if f.endswith('.npy')])
                        if not ignore_all and dataset not in datasets_that_need_ignore_masking:
                            self.ignore_mask_files.extend([None for f in os.listdir(f'{dataset_path}/ignore_masks') if f.endswith('.npy')])
                        else:
                            self.ignore_mask_files.extend([f'{dataset_path}/ignore_masks/{f}' for f in np.sort(os.listdir(f'{dataset_path}/ignore_masks')) if f.endswith('.npy')])

                        if dataset in silver_dataset:
                            self.example_weights.extend([silver_data_weight for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])
                        else:
                            self.example_weights.extend([1.0 for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])

        

    def __len__(self):
        return len(self.img_files)

    def _resize(self, img, ann, ignore_mask, flows_dx, flows_dy, cell_prob):
        #convert to image grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #resize if needed:
        if img.shape[0] < self.crop_size and img.shape[1] < self.crop_size:
            if img.shape[0] < img.shape[1]:
                multiplier = self.crop_size / img.shape[0]
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                flows_dx = cv2.resize(flows_dx, (int(multiplier * flows_dx.shape[1]) + 1, int(multiplier * flows_dx.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                flows_dy = cv2.resize(flows_dy, (int(multiplier * flows_dy.shape[1]) + 1, int(multiplier * flows_dy.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                cell_prob = cv2.resize(cell_prob, (int(multiplier * cell_prob.shape[1]) + 1, int(multiplier * cell_prob.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            else:
                multiplier = self.crop_size / img.shape[1]
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                flows_dx = cv2.resize(flows_dx, (int(multiplier * flows_dx.shape[1]) + 1, int(multiplier * flows_dx.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                flows_dy = cv2.resize(flows_dy, (int(multiplier * flows_dy.shape[1]) + 1, int(multiplier * flows_dy.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                cell_prob = cv2.resize(cell_prob, (int(multiplier * cell_prob.shape[1]) + 1, int(multiplier * cell_prob.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
        elif img.shape[0] < self.crop_size:
            multiplier = self.crop_size / img.shape[0]
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            flows_dx = cv2.resize(flows_dx, (int(multiplier * flows_dx.shape[1]) + 1, int(multiplier * flows_dx.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            flows_dy = cv2.resize(flows_dy, (int(multiplier * flows_dy.shape[1]) + 1, int(multiplier * flows_dy.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            cell_prob = cv2.resize(cell_prob, (int(multiplier * cell_prob.shape[1]) + 1, int(multiplier * cell_prob.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
        elif img.shape[1] < self.crop_size:
            multiplier = self.crop_size / img.shape[1]
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            flows_dx = cv2.resize(flows_dx, (int(multiplier * flows_dx.shape[1]) + 1, int(multiplier * flows_dx.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            flows_dy = cv2.resize(flows_dy, (int(multiplier * flows_dy.shape[1]) + 1, int(multiplier * flows_dy.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            cell_prob = cv2.resize(cell_prob, (int(multiplier * cell_prob.shape[1]) + 1, int(multiplier * cell_prob.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img, ann, ignore_mask, flows_dx, flows_dy, cell_prob

    def _preprocess(self, img, use_advanced_augmentations=False):
        if use_advanced_augmentations:
            if False:
                #Percentile Normalization
                p1, p99 = np.percentile(img, (1, 99))
                img = np.clip((img - p1) / (p99 - p1), 0, 1)

                #Z-Score Normalization
                # img = (img - img.mean()) / (img.std() + 1e-8)

                if isinstance(img, np.ndarray):
                    img = img.astype(np.float32)
                    #cvt to 3 channel
                    if len(img.shape) < 3 or img.shape[2] != 3:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img = img.to(torch.float32)
                    img = img.permute(1, 2, 0)
                    img = img.cpu().numpy()
                    #cvt to 3 channel
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            else:
                #adaptive hist normalization
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)
                img = cv2.normalize(img_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                #cvt to 3 channel
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            #Not Doing CLAHE
        else:
            #adaptive hist normalization
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)
            img = cv2.normalize(img_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            #cvt to 3 channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img
    
    def _data_augmentation(self, img, ignore_mask, flows_dx, flows_dy, cell_prob, use_advanced_augmentations=False):
        if use_advanced_augmentations:
            #prevent overflow
            img = torch.Tensor(img).to(torch.float32).permute(2, 0, 1)
            ignore_mask = torch.Tensor(ignore_mask).to(torch.float32).unsqueeze(0) #.cpu().numpy()
            flows_dx = torch.Tensor(flows_dx).to(torch.float32).unsqueeze(0) #.cpu().numpy()
            flows_dy = torch.Tensor(flows_dy).to(torch.float32).unsqueeze(0) #.cpu().numpy()
            cell_prob = torch.Tensor(cell_prob).to(torch.float32).unsqueeze(0) #.cpu().numpy()
            # cell_prob = np.where(cell_prob > 0.5, 1, 0)

            #geometric augmentations - apply to image and labels
            random_scale_aug = RandomScale(p=0.25, scale_range=(0.25, 4.0))
            rescaled_data = random_scale_aug([img, ignore_mask, flows_dx, flows_dy, cell_prob])

            img = rescaled_data[0]
            ignore_mask = rescaled_data[1].squeeze(0).cpu().numpy()
            ignore_mask = np.where(ignore_mask > 0.5, 1, 0)
            flows_dx = rescaled_data[2].squeeze(0).cpu().numpy()
            flows_dy = rescaled_data[3].squeeze(0).cpu().numpy()
            cell_prob = rescaled_data[4].squeeze(0).cpu().numpy()
            cell_prob = np.where(cell_prob > 0.5, 1, 0)

            intensity_augmentations = transforms.Compose(
                [
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
                    RandomInvertContrast(p=0.25)
                ]
            )

            # rand = random.random()
            # if rand <= 0.125:
            #     intensity_augmentations.transforms.append(PoissonNoise(lam_scale=30))
            # elif rand > 0.125 and rand <= 0.25:
            #     intensity_augmentations.transforms.append(transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)))
            # elif rand > 0.25 and rand <= 0.375:
            #     intensity_augmentations.transforms.append(RandomDownsampleUpsample(scale_range=(0.5, 0.9)))
            # elif rand > 0.375 and rand <= 0.5:
            #     intensity_augmentations.transforms.append(RandomSpeckleNoise(std=0.05))

            img = torch.clamp((img - img.min()) / (img.max() - img.min()), 0.0, 1.0)
            img = intensity_augmentations(img).cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
            img = self._preprocess(img, use_advanced_augmentations)

        else:
            #prevent overflow
            img = img.astype(np.float32)

            #random brightness
            alpha = random.uniform(0.95, 1.05)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

            # #random contrast / gamma
            # beta = random.randint(-1, 1)
            # gamma = random.uniform(0.9, 1.1)
            # img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
            # img = np.power(img, gamma)

            #back to uint8
            img = np.clip(img, 0, 255).astype(np.uint8)

            #randomly invert image
            if random.random() > 0.5:
                img = cv2.bitwise_not(img)

        return img, ignore_mask, flows_dx, flows_dy, cell_prob

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_files[idx]))
        ann = np.load(self.ann_files[idx])
        flows = np.load(self.flow_files[idx])
        flows_dx, flows_dy, cell_prob = flows[0], flows[1], flows[2]
        example_weight = self.example_weights[idx]
        if self.ignore_mask_files[idx] is not None:
            ignore_mask = np.load(self.ignore_mask_files[idx])
        else:
            ignore_mask = np.zeros_like(ann)

        #preprocess image
        img = self._preprocess(img, self.use_advanced_augmentations)

        if self.isTrain:
            #data augmentation
            img, ignore_mask, flows_dx, flows_dy, cell_prob = self._data_augmentation(img, ignore_mask, flows_dx, flows_dy, cell_prob, self.use_advanced_augmentations)
            ann = np.zeros_like(img)

        #resize everything
        img, ann, ignore_mask, flows_dx, flows_dy, cell_prob = self._resize(img, ann, ignore_mask, flows_dx, flows_dy, cell_prob)

        #random crop to size
        x = random.randint(0, img.shape[1] - self.crop_size)
        y = random.randint(0, img.shape[0] - self.crop_size)
        img = img[y:y+self.crop_size, x:x+self.crop_size]
        ann = ann[y:y+self.crop_size, x:x+self.crop_size]
        ignore_mask = ignore_mask[y:y+self.crop_size, x:x+self.crop_size]
        # dist_map = dist_map[y:y+self.crop_size, x:x+self.crop_size]
        flows_dx = flows_dx[y:y+self.crop_size, x:x+self.crop_size]
        flows_dy = flows_dy[y:y+self.crop_size, x:x+self.crop_size]
        cell_prob = cell_prob[y:y+self.crop_size, x:x+self.crop_size]

        #Invert Ignore Mask
        ignore_mask = np.where(ignore_mask > 0, self.ignore_mask_weight, 1)
    

        # prepare image and prompt for the model
        # inputs = self.processor(img, return_tensors="pt") #input image: shape: (256, 256, 3), range [0, 255]
        if self.use_advanced_augmentations and False: #CHANGE AND FALSE
            if self.crop_size != 896:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # make_normalize_transform(),
                        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
                    ]
                )
            else:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # make_normalize_transform()
                    ]
                )
        else:
            if self.crop_size != 896:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform(),
                        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
                    ]
                )
            else:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform()
                    ]
                )
        img = image_transform(img)
        inputs = {'pixel_values': img}

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        gt_mask_inputs = {}
        gt_mask_inputs['gt_mask'] = ann
        gt_mask_inputs['ignore_mask'] = ignore_mask
        # gt_mask_inputs['gt_dist_map'] = dist_map
        gt_mask_inputs['gt_flows_dx'] = flows_dx.astype(np.float32)
        gt_mask_inputs['gt_flows_dy'] = flows_dy.astype(np.float32)
        gt_mask_inputs['gt_cell_prob'] = cell_prob.astype(np.float32)
        gt_mask_inputs['example_weight'] = example_weight

        inputs['gt_data'] = gt_mask_inputs
        # inputs['binary_mask'] = label > 0.001

        return inputs




class DINODirectMaskDataset(Dataset):
    def __init__(self, dataset_root_path, dataset_list, split, include_silver_data, use_advanced_augmentations, ignore_all, ignore_mask_weight, silver_data_weight, crop_size):
        
        self.include_silver_data = include_silver_data
        self.use_advanced_augmentations = use_advanced_augmentations
        self.ignore_all = ignore_all
        self.ignore_mask_weight = ignore_mask_weight
        self.silver_data_weight = silver_data_weight
        self.crop_size = crop_size
        self.isTrain = 'train' == split

        self.img_files = []
        self.ann_files = []
        self.flow_files = []
        self.center_files = []
        self.ignore_mask_files = []
        self.example_weights = []

        for dataset in dataset_list:
            if dataset not in ood_datasets:
                # dataset_path = f'{dataset_root_path}/{dataset}/{split}'

                # self.img_files.extend([f'{dataset_path}/imgs/{f}' for f in np.sort(os.listdir(f'{dataset_path}/imgs')) if f.endswith('.png')])
                # self.ann_files.extend([f'{dataset_path}/anns/{f}' for f in np.sort(os.listdir(f'{dataset_path}/anns')) if f.endswith('.npy')])
                # if not ignore_all and dataset in datasets_that_need_ignore_masking:
                #     self.ignore_mask_files.extend([None for f in os.listdir(f'{dataset_path}/ignore_masks') if f.endswith('.npy')])
                # else:
                #     self.ignore_mask_files.extend([f'{dataset_path}/ignore_masks/{f}' for f in np.sort(os.listdir(f'{dataset_path}/ignore_masks')) if f.endswith('.npy')])

                # if dataset in silver_dataset:
                #     self.example_weights.extend([silver_data_weight for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])
                # else:
                #     self.example_weights.extend([1.0 for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])

                # USE ONCE 512 TRAINING DONE
                if not self.isTrain and dataset in silver_dataset:
                    continue
                else:
                    if not self.include_silver_data and dataset in silver_dataset:
                        continue
                    else:
                        dataset_path = f'{dataset_root_path}/{dataset}/{split}'

                        self.img_files.extend([f'{dataset_path}/imgs/{f}' for f in np.sort(os.listdir(f'{dataset_path}/imgs')) if f.endswith('.png')])
                        self.ann_files.extend([f'{dataset_path}/anns/{f}' for f in np.sort(os.listdir(f'{dataset_path}/anns')) if f.endswith('.npy')])
                        self.flow_files.extend([f'{dataset_path}/flows/{f}' for f in np.sort(os.listdir(f'{dataset_path}/flows')) if f.endswith('.npy')])
                        self.center_files.extend([f'{dataset_path}/centers/{f}' for f in np.sort(os.listdir(f'{dataset_path}/centers')) if f.endswith('.npy')])
                        if not ignore_all and dataset not in datasets_that_need_ignore_masking:
                            self.ignore_mask_files.extend([None for f in os.listdir(f'{dataset_path}/ignore_masks') if f.endswith('.npy')])
                        else:
                            self.ignore_mask_files.extend([f'{dataset_path}/ignore_masks/{f}' for f in np.sort(os.listdir(f'{dataset_path}/ignore_masks')) if f.endswith('.npy')])

                        if dataset in silver_dataset:
                            self.example_weights.extend([silver_data_weight for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])
                        else:
                            self.example_weights.extend([1.0 for f in os.listdir(f'{dataset_path}/imgs') if f.endswith('.png')])
                



    def __len__(self):
        return len(self.img_files)

    def _resize_centers(self, centers, size_x, size_y):
        center_indices = np.unique(centers)

        centers_new = np.zeros((size_x, size_y))

        for index in center_indices:
            if index != 0:
                coords = np.argwhere(centers == index)
                coord_x = coords[0][0] / centers.shape[0]
                coord_y = coords[0][1] / centers.shape[1]

                coord_x_new = int(coord_x * size_x)
                coord_y_new = int(coord_y * size_y)

                centers_new[coord_x_new, coord_y_new] = index

        return centers_new


    def _resize(self, img, ann, flows_dx, flows_dy, cell_prob, centers, ignore_mask):
        #convert to image grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #resize if needed:
        if img.shape[0] < self.crop_size and img.shape[1] < self.crop_size:
            if img.shape[0] < img.shape[1]:
                multiplier = self.crop_size / img.shape[0]
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                flows_dx = cv2.resize(flows_dx, (int(multiplier * flows_dx.shape[1]) + 1, int(multiplier * flows_dx.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                flows_dy = cv2.resize(flows_dy, (int(multiplier * flows_dy.shape[1]) + 1, int(multiplier * flows_dy.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                cell_prob = cv2.resize(cell_prob, (int(multiplier * cell_prob.shape[1]) + 1, int(multiplier * cell_prob.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                centers = self._resize_centers(centers, int(multiplier * centers.shape[0]) + 1, int(multiplier * centers.shape[1]) + 1)
                ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            else:
                multiplier = self.crop_size / img.shape[1]
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                flows_dx = cv2.resize(flows_dx, (int(multiplier * flows_dx.shape[1]) + 1, int(multiplier * flows_dx.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                flows_dy = cv2.resize(flows_dy, (int(multiplier * flows_dy.shape[1]) + 1, int(multiplier * flows_dy.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
                cell_prob = cv2.resize(cell_prob, (int(multiplier * cell_prob.shape[1]) + 1, int(multiplier * cell_prob.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
                centers = self._resize_centers(centers, int(multiplier * centers.shape[0]) + 1, int(multiplier * centers.shape[1]) + 1)
                ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
        elif img.shape[0] < self.crop_size:
            multiplier = self.crop_size / img.shape[0]
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            flows_dx = cv2.resize(flows_dx, (int(multiplier * flows_dx.shape[1]) + 1, int(multiplier * flows_dx.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            flows_dy = cv2.resize(flows_dy, (int(multiplier * flows_dy.shape[1]) + 1, int(multiplier * flows_dy.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            cell_prob = cv2.resize(cell_prob, (int(multiplier * cell_prob.shape[1]) + 1, int(multiplier * cell_prob.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            centers = self._resize_centers(centers, int(multiplier * centers.shape[0]) + 1, int(multiplier * centers.shape[1]) + 1)
            ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
        elif img.shape[1] < self.crop_size:
            multiplier = self.crop_size / img.shape[1]
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            flows_dx = cv2.resize(flows_dx, (int(multiplier * flows_dx.shape[1]) + 1, int(multiplier * flows_dx.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            flows_dy = cv2.resize(flows_dy, (int(multiplier * flows_dy.shape[1]) + 1, int(multiplier * flows_dy.shape[0]) + 1), interpolation=cv2.INTER_CUBIC)
            cell_prob = cv2.resize(cell_prob, (int(multiplier * cell_prob.shape[1]) + 1, int(multiplier * cell_prob.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)
            centers = self._resize_centers(centers, int(multiplier * centers.shape[0]) + 1, int(multiplier * centers.shape[1]) + 1)
            ignore_mask = cv2.resize(ignore_mask, (int(multiplier * ignore_mask.shape[1]) + 1, int(multiplier * ignore_mask.shape[0]) + 1), interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img, ann, flows_dx, flows_dy, cell_prob, centers, ignore_mask

    def _preprocess(self, img, use_advanced_augmentations=False):
        if use_advanced_augmentations:
            #Percentile Normalization
            p1, p99 = np.percentile(img, (1, 99))
            img = np.clip((img - p1) / (p99 - p1), 0, 1)

            #Z-Score Normalization
            # img = (img - img.mean()) / (img.std() + 1e-8)

            if isinstance(img, np.ndarray):
                img = img.astype(np.float32)
                #cvt to 3 channel
                if len(img.shape) < 3 or img.shape[2] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = img.to(torch.float32)
                img = img.permute(1, 2, 0)
                img = img.cpu().numpy()
                #cvt to 3 channel
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

            #Not Doing CLAHE
        else:
            #adaptive hist normalization
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)
            img = cv2.normalize(img_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            #cvt to 3 channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img
    
    def _data_augmentation(self, img, ignore_mask, flows_dx, flows_dy, cell_prob, ann, centers, use_advanced_augmentations=False):
        if use_advanced_augmentations:
            #prevent overflow
            img = img.to(torch.float32).permute(2, 0, 1)
            ignore_mask = torch.Tensor(ignore_mask).to(torch.float32).unsqueeze(0)
            flows_dx = torch.Tensor(flows_dx).to(torch.float32).unsqueeze(0).cpu().numpy()
            flows_dy = torch.Tensor(flows_dy).to(torch.float32).unsqueeze(0).cpu().numpy()
            cell_prob = torch.Tensor(cell_prob).to(torch.float32).unsqueeze(0).cpu().numpy()
            cell_prob = np.where(cell_prob > 0.5, 1, 0)

            #geometric augmentations - apply to image and labels
            rand_num = random.random()
            if rand_num < 0.25:
                rand_scale = (random.random() * 3.75) + 0.25

                img = cv2.resize(img, (int(rand_scale * img.shape[0]) + 1, int(rand_scale * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
                ann = cv2.resize(ann, (int(rand_scale * ann.shape[0]) + 1, int(rand_scale * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
                flows_dx = cv2.resize(flows_dx, (int(rand_scale * flows_dx.shape[0]) + 1, int(rand_scale * flows_dx.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
                flows_dy = cv2.resize(flows_dy, (int(rand_scale * flows_dy.shape[0]) + 1, int(rand_scale * flows_dy.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
                cell_prob = cv2.resize(cell_prob, (int(rand_scale * cell_prob.shape[0]) + 1, int(rand_scale * cell_prob.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
                cell_prob = np.where(cell_prob > 0.5, 1, 0)
                centers = self._resize_centers(centers, int(rand_scale * centers.shape[0]) + 1, int(rand_scale * centers.shape[1]) + 1)
                ignore_mask = cv2.resize(ignore_mask, (int(rand_scale * ignore_mask.shape[0]) + 1, int(rand_scale * ignore_mask.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)

            intensity_augmentations = transforms.Compose(
                [
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
                    RandomInvertContrast(p=0.25)
                ]
            )

            rand = random.random()
            if rand <= 0.125:
                intensity_augmentations.transforms.append(PoissonNoise(lam_scale=30))
            elif rand > 0.125 and rand <= 0.25:
                intensity_augmentations.transforms.append(transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)))
            elif rand > 0.25 and rand <= 0.375:
                intensity_augmentations.transforms.append(RandomDownsampleUpsample(scale_range=(0.5, 0.9)))
            elif rand > 0.375 and rand <= 0.5:
                intensity_augmentations.transforms.append(RandomSpeckleNoise(std=0.05))

            img = torch.clamp((img - img.min()) / (img.max() - img.min()), 0.0, 1.0)
            img = intensity_augmentations(img)
            img = self._preprocess(img, use_advanced_augmentations)
            img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
        else:
            #prevent overflow
            img = img.astype(np.float32)

            #random brightness
            alpha = random.uniform(0.95, 1.05)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

            # #random contrast / gamma
            # beta = random.randint(-1, 1)
            # gamma = random.uniform(0.9, 1.1)
            # img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
            # img = np.power(img, gamma)

            #back to uint8
            img = np.clip(img, 0, 255).astype(np.uint8)

            #randomly invert image
            if random.random() > 0.5:
                img = cv2.bitwise_not(img)

        return img, ignore_mask, flows_dx, flows_dy, cell_prob, ann, centers

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_files[idx]))
        ann = np.load(self.ann_files[idx])
        flows = np.load(self.flow_files[idx])
        flows_dx, flows_dy, cell_prob = flows[0], flows[1], flows[2]
        centers = np.load(self.center_files[idx])
        example_weight = self.example_weights[idx]
        if self.ignore_mask_files[idx] is not None:
            ignore_mask = np.load(self.ignore_mask_files[idx])
        else:
            ignore_mask = np.zeros_like(ann)

        # plt.imshow(img)
        # plt.savefig('img_before.png')

        #preprocess image
        img = self._preprocess(img, self.use_advanced_augmentations)

        if self.isTrain:
            #data augmentation
            img, ignore_mask, flows_dx, flows_dy, cell_prob, ann, centers = self._data_augmentation(img, ignore_mask, flows_dx, flows_dy, cell_prob, ann, centers, self.use_advanced_augmentations)
        
        #resize everything
        img, ann, flows_dx, flows_dy, cell_prob, centers, ignore_mask = self._resize(img, ann, flows_dx, flows_dy, cell_prob, centers, ignore_mask)

        #random crop to size
        x = random.randint(0, img.shape[1] - self.crop_size)
        y = random.randint(0, img.shape[0] - self.crop_size)
        img = img[y:y+self.crop_size, x:x+self.crop_size]
        ann = ann[y:y+self.crop_size, x:x+self.crop_size]
        flows_dx = flows_dx[y:y+self.crop_size, x:x+self.crop_size]
        flows_dy = flows_dy[y:y+self.crop_size, x:x+self.crop_size]
        cell_prob = cell_prob[y:y+self.crop_size, x:x+self.crop_size]
        centers = centers[y:y+self.crop_size, x:x+self.crop_size]
        ignore_mask = ignore_mask[y:y+self.crop_size, x:x+self.crop_size]
        
        #Invert Ignore Mask
        ignore_mask = np.where(ignore_mask > 0, 0, 1)
        
        # prepare image and prompt for the model
        # inputs = self.processor(img, return_tensors="pt") #input image: shape: (256, 256, 3), range [0, 255]
        if self.use_advanced_augmentations:
            if self.crop_size != 896:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # make_normalize_transform(),
                        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
                    ]
                )
            else:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # make_normalize_transform()
                    ]
                )
        else:
            if self.crop_size != 896:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform(),
                        transforms.Resize(896, interpolation=transforms.InterpolationMode.BICUBIC),
                    ]
                )
            else:
                image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        make_normalize_transform()
                    ]
                )
        img = image_transform(img)
        inputs = {'pixel_values': img}

        # plt.imshow(img.permute(1, 2, 0))
        # plt.savefig('img_after.png')

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        gt_mask_inputs = {}
        gt_mask_inputs['gt_mask'] = ann
        gt_mask_inputs['gt_flows_dx'] = flows_dx.astype(np.float32)
        gt_mask_inputs['gt_flows_dy'] = flows_dy.astype(np.float32)
        gt_mask_inputs['gt_cell_prob'] = cell_prob.astype(np.float32)
        gt_mask_inputs['gt_centers'] = centers
        gt_mask_inputs['ignore_mask'] = ignore_mask
        gt_mask_inputs['example_weight'] = example_weight

        inputs['gt_data'] = gt_mask_inputs
        # inputs['binary_mask'] = label > 0.001
        

        return inputs


