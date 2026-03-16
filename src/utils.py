import wandb
import torch
from torch import nn
import cv2
import numpy as np
import random
from skimage.segmentation import watershed
from cellpose.dynamics import compute_masks
import torch.nn.functional as F
import torchvision.transforms.functional as TTF
import torchvision.transforms.functional as TF
from torchvision.ops.boxes import batched_nms

from metrics.seg_det import get_seg_det_metrics
from metrics.max_matching_accuracy_and_shape_score import compute_mma_and_shape_scores
from metrics.cell_count_score import calculate_cell_count_score
from metrics.average_precision import calculate_average_precision

import matplotlib.pyplot as plt


def init_wandb():
    run = wandb.init(project="cellseg")
    return run
    
def log_wandb(run, current_step, learning_rate, loss):
    run.log({"lr": learning_rate, "loss": loss}, step=current_step)

def lr_lambda_with_warmup(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        # Linear warmup
        return float(current_step) / float(warmup_steps)
    else:
        # Example: Linear decay after warmup
        return max(0.0, float(total_steps - current_step) / float(total_steps - warmup_steps))


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

            # plt.imshow(ignore_mask)
            # plt.savefig('metrics_ignore_mask.png')
            # plt.close()

            pred_label = pred_label * ignore_mask
            gt_label = gt_label * ignore_mask

        # plt.imshow(pred_label)
        # plt.savefig('metrics_pred_label.png')
        # plt.close()

        # mma, shape_score = compute_mma_and_shape_scores(gt_seg=gt_label, samcell_seg=pred_label)
        mma = compute_mma_and_shape_scores(gt_seg=gt_label, samcell_seg=pred_label)
        # cell_count_ratio, cell_count_score = calculate_cell_count_score(gt_label, pred_label)
        AP_50 = calculate_average_precision(gt_label, pred_label)
        mmas.append(mma)
        # shape_scores.append(shape_score)
        # cell_count_ratios.append(cell_count_ratio)
        # cell_count_scores.append(cell_count_score)
        AP_50s.append(AP_50)

    # seg, det = get_seg_det_metrics(gt_label, pred_label)

    metrics = {
        'MMA': np.mean(mmas), 
        'AP@50': np.mean(AP_50s), 
        # 'SEG': seg,
        # 'DET': det, 
        # 'CCR': np.mean(cell_count_ratios), 
        # 'CCS': np.mean(cell_count_scores), 
        # 'ShapeScore': np.mean(shape_scores)
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

    # plt.imshow(flows_x)
    # plt.savefig('metrics_flows_x.png')
    # plt.close()
    # plt.imshow(flows_y)
    # plt.savefig('metrics_flows_y.png')
    # plt.close()

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



def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
        """
        Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
        an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
        """
        # torch.max below raises an error on empty inputs, just skip in this case
        if torch.numel(masks) == 0:
            return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

        # Normalize shape to CxHxW
        shape = masks.shape
        h, w = shape[-2:]
        if len(shape) > 2:
            masks = masks.flatten(0, -3)
        else:
            masks = masks.unsqueeze(0)

        # Get top and bottom edges
        in_height, _ = torch.max(masks, dim=-1)
        in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
        bottom_edges, _ = torch.max(in_height_coords, dim=-1)
        in_height_coords = in_height_coords + h * (~in_height)
        top_edges, _ = torch.min(in_height_coords, dim=-1)

        # Get left and right edges
        in_width, _ = torch.max(masks, dim=-2)
        in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
        right_edges, _ = torch.max(in_width_coords, dim=-1)
        in_width_coords = in_width_coords + w * (~in_width)
        left_edges, _ = torch.min(in_width_coords, dim=-1)

        # If the mask is empty the right edge will be to the left of the left edge.
        # Replace these boxes with [0, 0, 0, 0]
        empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
        out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
        out = out * (~empty_filter).unsqueeze(-1)

        # Return to original shape
        if len(shape) > 2:
            out = out.reshape(*shape[:-2], 4)
        else:
            out = out[0]

        return out

def convert_direct_mask_preds_to_segmentations(preds, iou_preds, crop_size, adaptive_thresh=0.01, iou_thresh=0.1, nms_thresh=0.99, min_area=15):
    if len(preds) == 1:
        mask_preds = preds[0].unsqueeze(0)
    elif len(preds) > 1:
        mask_preds = torch.stack(preds).to(preds[0].device)
    else:
        return torch.zeros((1, crop_size, crop_size))

    #blur masks to make smoother for decoding
    mask_preds = TTF.gaussian_blur(mask_preds, kernel_size=[5, 5], sigma=[2.0, 2.0])

    #binarize masks using adaptive threshold
    thresholds = (torch.where(mask_preds > 0.01, mask_preds, 0).sum(dim=(1, 2)) / torch.where(mask_preds > 0.01, 1, 0).sum(dim=(1, 2))) * adaptive_thresh
    binary_mask_preds = torch.where(mask_preds > thresholds.view(thresholds.shape[0], 1, 1), 1, 0)

    #filter out masks below predicted iou threshold
    valid_indices = torch.argwhere(torch.tensor(iou_preds) > iou_thresh).squeeze(1)
    valid_preds = binary_mask_preds[valid_indices]
    valid_iou_preds = torch.tensor(iou_preds)[valid_indices]
    

    #remove duplicate masks via nms
    mask_boxes = batched_mask_to_box(valid_preds)
    indices_to_keep = batched_nms(
        mask_boxes.float(),
        valid_iou_preds,
        torch.zeros_like(mask_boxes[:, 0]),  # categories
        iou_threshold=nms_thresh,
    )
    filtered_masks = valid_preds[indices_to_keep]
    valid_original_indices = valid_indices[indices_to_keep]
    valid_iou_preds = torch.tensor(iou_preds)[valid_original_indices]

    #remove small masks and small holes
    cleaned_masks = []
    for mask in filtered_masks:
        mask = remove_small_regions(mask.detach().cpu().numpy(), min_area, mode="holes")
        mask = remove_small_regions(mask, min_area, mode="islands")
        cleaned_masks.append(torch.tensor(mask))

    if len(cleaned_masks) > 1:
        cleaned_mask_scores = torch.stack(cleaned_masks) * mask_preds[valid_original_indices]
    elif len(cleaned_masks) == 1:
        return cleaned_masks[0].unsqueeze(0)
    else:
        return torch.zeros((1, crop_size, crop_size))


    background_layer = torch.ones(cleaned_mask_scores[0].shape).unsqueeze(0).to(mask_preds.device) * 1e-5
    filtered_preds = torch.cat([background_layer, cleaned_mask_scores])
    non_overlapping_segmentations = torch.argmax(filtered_preds, axis=0).detach().cpu().numpy()

    return non_overlapping_segmentations

def get_direct_mask_metrics(objectness_preds, iou_preds, gt_labels, ignore_masks=None, crop_size=256):
    pred_labels = convert_direct_mask_preds_to_segmentations(objectness_preds[0], iou_preds[0], crop_size=crop_size)

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
#  SOFT DICE LOSS
#####################
class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Soft Dice Loss for binary segmentation.
        
        Args:
            eps (float): A small constant for numerical stability to avoid 
                            division by zero.
        """
        super(SoftDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        """
        Calculates the soft Dice loss.

        Args:
            preds (torch.Tensor): The model's predictions (probabilities).
            targets (torch.Tensor): The ground truth labels (binary, 0 or 1).
        
        Returns:
            torch.Tensor: The computed loss value.
        """
        # Flatten the preds and targets for easier computation
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice_coefficient = (2. * intersection + self.eps) / (preds.sum() + targets.sum() + self.eps)
        
        # Dice loss is 1 - Dice coefficient
        loss = 1.0 - dice_coefficient
        
        return loss


#####################
#        IoU
#####################
def get_IoU(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """
    Calculates the Intersection over Union (IoU) metric for binary segmentation masks

    Args:
        preds (torch.Tensor): Predicted masks (e.g., after a sigmoid and thresholding), 
                                 binary tensor of shape (B, H, W) or (B, 1, H, W), etc.
        targets (torch.Tensor): Ground truth masks, binary tensor of the same shape as preds.
        eps (float): A small value to prevent division by zero.

    Returns:
        float: The IoU score as a Python float.
    """
    # Use int or float tensors for the operations
    preds = preds.int()
    targets = targets.int()
    
    # Flatten the spatial dimensions for easier sum
    # Shape: (B, H*W)
    preds_flat = preds.flatten(start_dim=-2)
    targets_flat = targets.flatten(start_dim=-2)

    # Calculate intersection (logical AND)
    intersection = (preds_flat & targets_flat).float().sum(dim=-1)

    # Calculate union (logical OR)
    union = (preds_flat | targets_flat).float().sum(dim=-1)

    # Handle cases where the union is 0 to avoid division by zero
    # Add a small epsilon to prevent errors
    iou = (intersection + eps) / (union + eps)

    return iou


#####################
#   MASK CLEANING
#####################
def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask


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

