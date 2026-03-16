import torch
from torch import nn
import cv2
import numpy as np
import random
from skimage.segmentation import watershed
from cellpose.dynamics import compute_masks
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from metrics.max_matching_accuracy_and_shape_score import compute_mma_and_shape_scores
from metrics.average_precision import calculate_average_precision

import matplotlib.pyplot as plt


def convert_label_to_rainbow(label):
    label_rainbow = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cell in np.unique(label):
        if cell == 0:
            continue #background
        label_rainbow[label == cell] = np.clip(np.random.rand(3) * 255, a_min=10, a_max=255)

    return label_rainbow


def get_metrics(gt_labels, pred_labels, ignore_masks=None):
    mmas = []
    shape_scores = []
    cell_count_ratios = []
    cell_count_scores = []
    AP_50s = []
    for i in range(len(pred_labels)):
        pred_label = pred_labels[i]
        gt_label = gt_labels[i]

        if ignore_masks is not None:
            if torch.is_tensor(ignore_masks[i]):
                ignore_mask = np.where(ignore_masks[i].detach().cpu().numpy() == 1, 1, 0)
            else:
                ignore_mask = np.where(ignore_masks[i] == 1, 1, 0)

            pred_label = pred_label * ignore_mask
            gt_label = gt_label * ignore_mask

        mma = compute_mma_and_shape_scores(gt_seg=gt_label, samcell_seg=pred_label)
        AP_50 = calculate_average_precision(gt_label, pred_label)
        mmas.append(mma)
        AP_50s.append(AP_50)

    # seg, det = get_seg_det_metrics(gt_label, pred_label)

    metrics = {
        'MMA': np.mean(mmas), 
        'AP@50': np.mean(AP_50s)
    }

    return metrics


def cells_from_dist_map(dist_maps):
    pred_labels = []
    for dist_map in dist_maps:
        cells_max = dist_map > 0.5
        cell_fill = dist_map > 0.05
        #find centroids of connected components
        contours, _ = cv2.findContours(cells_max.astype(np.uint8), 0, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(dist_map.shape, dtype=np.int32)

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

        pred_labels.append(labels)

    return pred_labels


def get_dist_map_metrics(pred_dist_maps, gt_labels, ignore_masks=None):
    pred_labels = cells_from_dist_map(pred_dist_maps)

    return get_metrics(gt_labels, pred_labels, ignore_masks)


def cells_from_flows(flows_x, flows_y, cell_prob, cellprob_threshold):
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
        flow_threshold=0,
        do_3D=False,
        min_size=15,
        max_size_fraction=0.4,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    return mask


def get_flows_metrics(pred_flows_x, pred_flows_y, pred_cell_prob, gt_labels, ignore_masks=None):
    pred_labels = cells_from_flows(pred_flows_x.squeeze(0), pred_flows_y.squeeze(0), pred_cell_prob.squeeze(0), cellprob_threshold=0.5)

    if len(pred_labels.shape) == 2:
        pred_labels = np.array([pred_labels])

    return get_metrics(gt_labels, pred_labels, ignore_masks)


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


#####################
#  AUGMENTATIONS
#####################

class RandomScale:
    def __init__(self, p=0.25, scale_range=(0.25, 4.0)):
        self.p = p
        self.scale_min, self.scale_max = scale_range

    def __call__(self, inputs):
        if random.random() < self.p:
            s = random.uniform(self.scale_min, self.scale_max)
            H, W = inputs[0].shape[-2:]
            new_h, new_w = int(H * s), int(W * s)

            rescaled_inputs = []
            for input in inputs:
                rescaled_inputs.append(TF.resize(input, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR))

            return rescaled_inputs
        return inputs



class RandomInvertContrast:
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            # assuming image is in [0,1]
            x = 1.0 - x

        return x


class PoissonNoise:
    def __init__(self, lam_scale=30):
        """
        p: probability of adding Poisson noise
        lam_scale: controls noise strength.
                   Higher = more noise (lambda multiplier).
        """
        self.lam_scale = lam_scale

    def __call__(self, image):

        # image is assumed to be in [0,1]
        # Scale up so Poisson sampling has a meaningful lambda
        scaled = image * self.lam_scale

        # Poisson noise
        noisy = torch.poisson(scaled)

        # Re-normalize to [0,1]
        noisy = noisy / self.lam_scale

        # Clamp to valid range
        image = torch.clamp(noisy, 0.0, 1.0)

        return image


class RandomDownsampleUpsample:
    def __init__(self, scale_range=(0.5, 0.9)):
        self.scale_range = scale_range

    def __call__(self, image):
        scale = random.uniform(*self.scale_range)
        c, h, w = image.shape
        # downsample
        new_h, new_w = int(h * scale), int(w * scale)
        image_small = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)
        # upsample
        image = F.interpolate(image_small, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)

        return image


class RandomSpeckleNoise:
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, image):
        noise = torch.randn_like(image) * self.std
        image = image + image * noise
        image = torch.clamp(image, 0.0, 1.0)
        return image




pc_bf_non_tissue_whole_cell_only_dataset = [
    #hand segmented vs auto segmented (weak label training), whole cell/nuclei/cyto, tissue/non-tissue, image type (pc_bf, fluor, etc), needs ignore masking, splits available
    'BCCD', #hand segmented, whole cell, non-tissue, pc_bf (stained), no ignore masking, train/test
    'cellpose_cyto/pc_bf_df_images', #hand segmented, whole cell, non-tissue, pc_bf, no ignore masking, train/test
    'DeepBacs/DeepBacs_Data_Segmentation_E.coli_Brightfield_dataset', #hand segmented, whole cell, non-tissue, pc_bf, no ignore masking, train/test
    'DeepBacs/DeepBacs_Data_Segmentation_Ecoli_stationary_phase', #hand segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train/test
    'DeepBacs/DeepBacs_Data_Segmentation_Staph_Aureus_dataset/brightfield_dataset', #hand segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train/test
    'LiveCell', #hand segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train/test
    'NeurIPS_2022_Challenge/pc_bf_df_images', # hand segmented, whole_cell, non-tissue, pc_bf, needs ignore masking, train/test
    'Omnipose/bact_phase', #hand segmented, whole cell, non-tissue, pc_bf, no ignore masking, train/test
    'Omnipose/worm', #hand segmented, whole cell/worm, non-tissue, pc_bf, no ignore masking, train/test
    'Omnipose/worm_high_res', #hand segmented, whole cell/worm, non-tissue, pc_bf, no ignore masking, train/test
    'YeaZ', #hand segmented, whole_cell, non-tissue, pc_bf, no ignore masking, train
    'BBBC030', #hand segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train
    'cpg0016-jump/source_1/pc_bf', #auto segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train
    'cpg0016-jump/source_3/pc_bf', #auto segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train
    'cpg0016-jump/source_4/pc_bf', #auto segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train
    'cpg0016-jump/source_5/pc_bf', #auto segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train
    'cpg0016-jump/source_6/pc_bf', #auto segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train
    'cpg0016-jump/source_11/pc_bf', #auto segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train
    'cpg0016-jump/source_15/pc_bf', #auto segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train
    'idr0095', #hand segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train
    'Ntera_2', #hand segmented, whole cell, non-tissue, pc_bf, needs ignore masking, train
    'TYC_Dataset' #hand segmented, whole cell, non-tissue, pc_bf, no ignore masking, train/test/val
]

