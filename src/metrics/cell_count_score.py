import numpy as np


def calculate_cell_count_score(gt_anns, model_out_anns):
    num_gt_masks = len(np.unique(gt_anns)) - 1
    num_out_masks = len(np.unique(model_out_anns)) - 1
    
    try:
        ratio = (num_out_masks / num_gt_masks)
        score = np.abs(num_gt_masks - num_out_masks) / num_gt_masks
        return ratio, score
    except:
        return 0.0, 1.0

    