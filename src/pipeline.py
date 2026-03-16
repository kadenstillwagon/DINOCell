from skimage.segmentation import watershed
from cellpose.dynamics import compute_masks
import math
import torch
from torch import nn
from torchvision.ops.boxes import batched_nms
import torchvision.transforms.functional as F
import numpy as np
import cv2
from slidingWindow import SlidingWindowHelper
from torchvision import transforms
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dinov2.dinov2.data.transforms import make_normalize_transform

def get_pipeline(objective_type, model, device, crop_size=256, use_advanced_augmentations=False, overlap_size=32):
    if objective_type == 'distance_map':
        return DINODistanceMapSlidingWindowPipeline(model, device, crop_size, use_advanced_augmentations)
    elif objective_type == 'directional_distance_map':
        return DINODistanceMapSlidingWindowPipeline(model, device, crop_size, use_advanced_augmentations)
    elif objective_type == 'flows':
        return DINOFlowsSlidingWindowPipeline(model, device, crop_size, use_advanced_augmentations, overlap_size)



class DINODistanceMapSlidingWindowPipeline:
    def __init__(self, model, device, crop_size=256, use_advanced_augmentations=False):
        self.model = model
        self.device = device
        self.crop_size = crop_size
        self.use_advanced_augmentations = use_advanced_augmentations
        self.sigmoid = nn.Sigmoid()
        self.sliding_window_helper = SlidingWindowHelper(crop_size, 32)
    
    def set_model(self, model):
        self.model = model

    def _resize(self, img, ann=None):

        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        #convert to grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #resize if needed:
        if img.shape[0] < self.crop_size and img.shape[1] < self.crop_size:
            if img.shape[0] < img.shape[1]:
                multiplier = self.crop_size / img.shape[0]
                if ann is not None:
                    ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
            else:
                multiplier = self.crop_size / img.shape[1]
                if ann is not None:
                    ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
        elif img.shape[0] < self.crop_size:
            multiplier = self.crop_size / img.shape[0]
            if ann is not None:
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
        elif img.shape[1] < self.crop_size:
            multiplier = self.crop_size / img.shape[1]
            if ann is not None:
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)

        return img, ann

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

        #default preprocessing
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
        img = image_transform(img).unsqueeze(0)
        return img.to(self.device)

    def get_model_prediction(self, image):
        image_orig = image.copy()
        image = self._preprocess(image_orig, self.use_advanced_augmentations)
        self.model.eval().to(self.device)

        # forward pass
        with torch.no_grad():
            preds = self.model(x=image)

        return preds

    def spilt_into_crops(self, image_orig):
        crops = []
        #split into crops
        for i in range(0, math.ceil(image_orig.shape[0] / (self.crop_size)) + 1):
            for j in range(0, math.ceil(image_orig.shape[1] / self.crop_size) + 1):
                min_x = i * self.crop_size
                min_y = j * self.crop_size
                min_x = min(min_x, image_orig.shape[0] - self.crop_size)
                min_y = min(min_y, image_orig.shape[1] - self.crop_size)
                crops.append((image_orig[min_x:min_x+self.crop_size, min_y:min_y+self.crop_size], (min_x, min_y)))

        return crops

    def predict_on_full_img(self, image_orig):
        orig_shape = image_orig.shape
        #rescale to 1000 (biggest dimension)
        if image_orig.shape[0] > image_orig.shape[1]:
            new_size = (int(image_orig.shape[1] * (1000 / image_orig.shape[0])), 1000)
        else:
            new_size = (1000, int(image_orig.shape[0] * (1000 / image_orig.shape[1])))

        # image_orig = cv2.resize(image_orig, (new_size))

        # crops = self.spilt_into_crops(image_orig)
        crops, orig_regions, crop_unique_region = self.sliding_window_helper.seperate_into_crops(image_orig)

        #predict on crops
        preds = []
        for crop in crops:
            crop_preds = self.get_model_prediction(crop)
            all_crop_preds = []
            for pred in crop_preds:
                pred = pred[0].cpu().detach().numpy()

                #resize to crop size
                pred = cv2.resize(pred, (self.crop_size, self.crop_size))
                all_crop_preds.append(pred)
            
            preds.append(all_crop_preds)

        preds = np.array(preds)

        #EDIT HERE
        image_preds = []
        for i in range(len(preds[0])):
            image_pred = self.sliding_window_helper.combine_crops(orig_shape, preds[:, i], orig_regions, crop_unique_region)
            image_preds.append(image_pred)

        return image_preds

    def cells_from_dist_map(self, dist_map):
        cells_max = dist_map > 0.5
        cell_fill = dist_map > 0.05
        #find centroids of connected components
        contours, _ = cv2.findContours(cells_max.astype(np.uint8), 0, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(dist_map.shape, dtype=np.int32)
        # for i, contour in enumerate(contours):
        #     contour = np.flip(contour, axis=2)
        #     mask[tuple(contour.T)] = i + 1

        for i, contour in enumerate(contours):
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # Handle cases where the moment is zero to avoid division by zero
                cX, cY = 0, 0

            #set closest pixel to centroid
            mask[int(cY), int(cX)] = i + 1


        labels = watershed(-dist_map, mask, mask=cell_fill).astype(np.int32)

        return labels
    
    def run(self, image, ann=None, return_raw_preds=False):
        image, ann = self._resize(image, ann)
        preds = self.predict_on_full_img(image)
        #EDIT HERE
        labels = self.cells_from_dist_map(preds[0])

        if return_raw_preds:
            if ann is not None:
                return labels, preds, image, ann
            else:
                return labels, preds, image
        if ann is not None:
            return labels, image, ann
        else:
            return labels, image


class DINOFlowsSlidingWindowPipeline:
    def __init__(self, model, device, crop_size=256, use_advanced_augmentations=False, overlap_size=32):
        self.model = model
        self.device = device
        self.crop_size = crop_size
        self.sigmoid = nn.Sigmoid()
        self.sliding_window_helper = SlidingWindowHelper(crop_size, overlap_size)
    
    def set_model(self, model):
        self.model = model

    def _resize(self, img, ann=None):
        #convert to grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #resize if needed:
        if img.shape[0] < self.crop_size and img.shape[1] < self.crop_size:
            if img.shape[0] < img.shape[1]:
                multiplier = self.crop_size / img.shape[0]
                if ann is not None:
                    ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
            else:
                multiplier = self.crop_size / img.shape[1]
                if ann is not None:
                    ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
                img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
        elif img.shape[0] < self.crop_size:
            multiplier = self.crop_size / img.shape[0]
            if ann is not None:
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)
        elif img.shape[1] < self.crop_size:
            multiplier = self.crop_size / img.shape[1]
            if ann is not None:
                ann = cv2.resize(ann, (int(multiplier * ann.shape[1]) + 1, int(multiplier * ann.shape[1]) + 1), interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, (int(multiplier * img.shape[1]) + 1, int(multiplier * img.shape[1]) + 1), interpolation=cv2.INTER_CUBIC)

        return img, ann

    def _preprocess(self, img):
        #adaptive hist normalization
        img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)
        img = cv2.normalize(img_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        #cvt to 3 channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        #default preprocessing
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
        img = image_transform(img).unsqueeze(0)
        return img.to(self.device)

    def get_model_prediction(self, image):
        image_orig = image.copy()
        image = self._preprocess(image_orig)
        self.model.eval().to(self.device)

        # forward pass
        with torch.no_grad():
            preds = self.model(x=image)

        return preds

    def spilt_into_crops(self, image_orig):
        crops = []
        #split into crops
        for i in range(0, math.ceil(image_orig.shape[0] / (self.crop_size)) + 1):
            for j in range(0, math.ceil(image_orig.shape[1] / self.crop_size) + 1):
                min_x = i * self.crop_size
                min_y = j * self.crop_size
                min_x = min(min_x, image_orig.shape[0] - self.crop_size)
                min_y = min(min_y, image_orig.shape[1] - self.crop_size)
                crops.append((image_orig[min_x:min_x+self.crop_size, min_y:min_y+self.crop_size], (min_x, min_y)))

        return crops

    def predict_on_full_img(self, image_orig):
        orig_shape = image_orig.shape
        #rescale to 1000 (biggest dimension)
        if image_orig.shape[0] > image_orig.shape[1]:
            new_size = (int(image_orig.shape[1] * (1000 / image_orig.shape[0])), 1000)
        else:
            new_size = (1000, int(image_orig.shape[0] * (1000 / image_orig.shape[1])))

        # image_orig = cv2.resize(image_orig, (new_size))

        # crops = self.spilt_into_crops(image_orig)
        # crops, orig_regions, crop_unique_region = self.sliding_window_helper.seperate_into_crops(image_orig) #CHANGE
        crops, orig_regions = self.sliding_window_helper.seperate_into_crops_v2(image_orig)

        #predict on crops
        preds = []
        for crop in crops:
            crop_preds = self.get_model_prediction(crop)
            all_crop_preds = []
            for pred in crop_preds:
                pred = pred[0].cpu().detach().numpy()

                #resize to crop size
                pred = cv2.resize(pred, (self.crop_size, self.crop_size))
                all_crop_preds.append(pred)
            
            preds.append(all_crop_preds)

        preds = np.array(preds)

        #EDIT HERE
        image_preds = []
        for i in range(len(preds[0])):
            # image_pred = self.sliding_window_helper.combine_crops(orig_shape, preds[:, i], orig_regions, crop_unique_region) #CHANGE
            image_pred = self.sliding_window_helper.combine_crops_v2(orig_shape, preds[:, i], orig_regions)
            image_preds.append(image_pred)

        return image_preds

    def cells_from_flows(self, flows_x, flows_y, cell_prob, flow_threshold, cellprob_threshold):
        """
        flows_3ch => shape (3, H, W) => (dx, dy, cell_prob).
        We interpret channel 0 => dx, 1 => dy, 2 => cell_prob
        Then we call cellpose's compute_masks(dP, cellprob_threshold=whatever).
        """

        # assemble dP with shape (2,H,W)
        dP = np.stack([flows_y, flows_x], axis=0)

        mask = compute_masks(
            dP=dP,
            cellprob=cell_prob,
            niter=250,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            do_3D=False,
            min_size=15,
            max_size_fraction=0.4,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        return mask
    
    def run(self, image, ann=None, return_raw_preds=False):
        image, ann = self._resize(image, ann)
        preds = self.predict_on_full_img(image)
        #EDIT HERE
        labels = self.cells_from_flows(preds[0], preds[1], preds[2], flow_threshold=0, cellprob_threshold=0.5)

        if return_raw_preds:
            if ann is not None:
                return labels, preds, image, ann
            else:
                return labels, preds, image
        if ann is not None:
            return labels, image, ann
        else:
            return labels, image



