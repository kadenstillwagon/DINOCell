import numpy as np

def calculate_average_precision(gt_anns, model_out_anns, iou_threshold=0.5):
  try:
    num_masks = len(np.unique(model_out_anns)) - 1
    TP = get_true_positives(gt_anns, model_out_anns, iou_threshold)
    FP = num_masks - TP
    FN = len(np.unique(gt_anns)) - TP

    precision = TP / (TP + FP + FN)
    # precision = TP / (TP + FP)

    return precision
  except:
    return 0.0


def get_true_positives(gt_anns, model_out_anns, iou_threshold):
  TP = 0

  model_anns = np.unique(model_out_anns)[1:]

  for label in model_anns:
    mask_locations = np.argwhere(model_out_anns == label)
    gt_anns_at_mask = gt_anns[mask_locations[:, 0], mask_locations[:, 1]]
    gt_intersection_labels, gt_intersection_label_counts = np.unique(gt_anns_at_mask, return_counts=True)
    for i in range(len(gt_intersection_labels)):
      model_ann_size = len(mask_locations)
      gt_label_size = len(np.argwhere(gt_anns == gt_intersection_labels[i]))

      iou = gt_intersection_label_counts[i] / (model_ann_size + gt_label_size - gt_intersection_label_counts[i])

      if iou >= iou_threshold:
        TP += 1
        break

  return TP


