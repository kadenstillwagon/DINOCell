import numpy as np
import copy
import networkx as nx
import sys
# from parameter_calculations import get_segmentation_parameters_for_shape_score


#DENOMINATOR

def calculate_denominator(gt_seg, samcell_seg):
  """
  Calculates the denominator of accuracy metric as the union between the area
  covered by the ground truth segmenetations and the area covered by SAMCell's
  segmentations
  Author(s): ####

  args:
      gt_seg (np.ndarray): Array of shape (W,H) representing the ground truth segmentation.
      samcell_seg (np.ndarray): Array of shape (W,H) representing SAMCell's segmentation.

  Returns:
      denominator (int): Integer representing the number of pixels coverd by the
                          union between the GT and SAMCell segmentations

  """

  #Get segmentation indices
  gt_seg_indices = np.argwhere(gt_seg > 0)
  samcell_seg_indices = np.argwhere(samcell_seg > 0)

  #Convert to sets
  gt_seg_indices_set = set(tuple(row) for row in gt_seg_indices)
  samcell_seg_indices_set = set(tuple(row) for row in samcell_seg_indices)

  #Calculate Union
  union_gt_samcell_seg_indices = gt_seg_indices_set | samcell_seg_indices_set

  return len(union_gt_samcell_seg_indices)



#NUMERATOR

def create_overlap_dict_max_matching(gt_seg, samcell_seg):
  #Same as greedy approach but doesn't make best_segmentation_index_dict or
  #keep track of Best Segmentation Index for each GT cell
  overlap_dict = {}

  gt_seg_labels = np.unique(gt_seg)[1:]

  for label in gt_seg_labels:
    overlap_dict[f'GT_{int(label)}'] = {}
    mask_locations = np.argwhere(gt_seg == label)
    
    if mask_locations.shape[1] == 2:
      samcell_labels_at_mask = samcell_seg[mask_locations[:, 0], mask_locations[:, 1]] 
    else:
      samcell_labels_at_mask = samcell_seg[mask_locations[0, :], mask_locations[1, :]]
    samcell_labels, samcell_label_counts = np.unique(samcell_labels_at_mask, return_counts=True)
    for i in range(len(samcell_labels)):
      if samcell_labels[i] != 0:
        overlap_dict[f'GT_{int(label)}'][f'OUT_{int(samcell_labels[i])}'] = int(samcell_label_counts[i])

  return overlap_dict


def get_max_overlap_matching(seg_overlap_dict):
  #Turn GT, model output segmentation matching into maximum weight bipartite matching problem
  #One set of nodes GT for ground truth segs, one set of nodes OUT for output segs
  #Edges between GT and OUT nodes are weight by their area of overlap
  #Max weight bipartite matching will return a matching between GT and OUT such
  #that each OUT segmentation can be assigned to at most 1 GT segmentation and
  #resulting matching will be the matching with highest overlap area

  #Create networkx graph
  G = nx.Graph()

  #Create edges between GT segs and every output seg that overlaps with overlap area as edge weight
  for gt_label, out_seg_overlaps in seg_overlap_dict.items():
    for out_seg_label, overlap_area in out_seg_overlaps.items():
      G.add_edge(gt_label, out_seg_label, weight=overlap_area)

  #Compute the maximum weight matching of G
    #Matching - subset of edges where no node occurs more that once
    #Weight of matching sum of the weights of its edges
    #Maximal matching cannot add more edges and still be a matchign
  max_overlap_matching = nx.max_weight_matching(G, maxcardinality=False)

  return max_overlap_matching


def calculate_total_area_of_associate_segmentation_overlaps_max_matching(seg_overlap_dict, segmentation_associations):
  total_area = 0

  for association in segmentation_associations:
    if 'GT' in association[0]:
      gt_label = association[0]
      out_seg_label = association[1]
    else:
      gt_label = association[1]
      out_seg_label = association[0]

    overlap_area = seg_overlap_dict[gt_label][out_seg_label]

    total_area += overlap_area

  return total_area


def calculate_numerator_max_matching(gt_seg, samcell_seg):

  #Calculate Overlap Dictionary
  segmentation_overlap_dict = create_overlap_dict_max_matching(
      gt_seg=gt_seg,
      samcell_seg=samcell_seg
  )

  #Associate GT and Output Segmentations (Max Matching)
  segmentation_associations = get_max_overlap_matching(
    seg_overlap_dict=copy.deepcopy(segmentation_overlap_dict)
  )

  #Calculate Total Overlap Area of GT and Output Segmentation
  numerator = calculate_total_area_of_associate_segmentation_overlaps_max_matching(
      seg_overlap_dict=segmentation_overlap_dict,
      segmentation_associations=segmentation_associations
  )

  return numerator, segmentation_associations


#ACCURACY

def compute_accuracy_max_matching(gt_seg, samcell_seg):

  #Calculate Denominator (Union of GT and output segmentations)
  denominator = calculate_denominator(
      gt_seg=gt_seg,
      samcell_seg=samcell_seg
  )

  #Calculate Numerator (Total Area of GT and Associated Segmentation Overlap)
  numerator, segmentation_associations = calculate_numerator_max_matching(
      gt_seg=gt_seg,
      samcell_seg=samcell_seg
  )

  mma = numerator / (denominator + 1e-5)

  return mma, segmentation_associations


#SHAPE SCORE
# def get_single_association_shape_similarity_score(gt_label, samcell_seg_label, gt_seg, samcell_seg, img):
#   """
#   Calculates a shape similarity score for a single model output, ground truth pair

#   Steps:
#     1) Computes shape parameters on GT and model output segmentations
#     2) Computes parameters similarity score: larger value / smaller value
#     3) Combines individual parameter similarity score into single shape similarity score
#       -Currently just the average of them

#   Author(s): ####

#   args:
#       gt_label (int): index/label of ground truth segmentation
#       samcell_seg_label (int): index/label of model output segmentation
#       gt_seg (np.ndarray): Array of shape (W,H) representing the ground truth segmentation.
#       samcell_seg (np.ndarray): Array of shape (W,H) representing SAMCell's segmentation.
#       img (np.ndarray): Array of shape (W,H) representing the grayscale image.

#   Returns:
#       similarity_score (float): score calculated from similarity of shape parameters between the segmentations

#   """

#   # Get ground truth segmentation shape parameters
#   gt_seg_shape_params = get_segmentation_parameters_for_shape_score(
#       segmentations=gt_seg,
#       seg_index=int(gt_label[3:]),
#       img=img
#   )

#   # Get model output segmentation shape parameters
#   samcell_seg_shape_params = get_segmentation_parameters_for_shape_score(
#       segmentations=samcell_seg,
#       seg_index=int(samcell_seg_label[4:]),
#       img=img
#   )

#   # Create Dictionary of Similarity Scores for each Shape Metric
#   shape_metric_similarity_scores = {}
#   for shape_metric in list(gt_seg_shape_params.keys()):
#     max_metric_value = np.max([gt_seg_shape_params[shape_metric], samcell_seg_shape_params[shape_metric]])
#     min_metric_value = np.min([gt_seg_shape_params[shape_metric], samcell_seg_shape_params[shape_metric]])

#     metric_similarity_score = min_metric_value / max_metric_value

#     shape_metric_similarity_scores[shape_metric] = metric_similarity_score


#   # Combine Shape Metric Similarity Scores into Single Similarity Score (Current: Take Average)
#   total_similarity_score = 0
#   for shape_metric in list(shape_metric_similarity_scores.keys()):
#     total_similarity_score += shape_metric_similarity_scores[shape_metric]

#   similarity_score = total_similarity_score / len(shape_metric_similarity_scores.keys())

#   return similarity_score


# def compute_shape_score(img, gt_seg, samcell_seg, segmentation_associations):
#   total_shape_similarity_score = 0
#   successful_single_association_calculations = 0

#   for association in segmentation_associations:
#     try:
#       if 'GT' in association[0]:
#         gt_label = association[0]
#         samcell_seg_label = association[1]
#       else:
#         gt_label = association[1]
#         samcell_seg_label = association[0]

#       shape_similarity_score = get_single_association_shape_similarity_score(
#           gt_label=gt_label,
#           samcell_seg_label=samcell_seg_label,
#           gt_seg=gt_seg,
#           samcell_seg=samcell_seg,
#           img=img
#       )

#       total_shape_similarity_score += shape_similarity_score
#       successful_single_association_calculations += 1
#     except:
#       continue

#   if successful_single_association_calculations > 0:
#     average_shape_similarity_score = total_shape_similarity_score / successful_single_association_calculations
#     return average_shape_similarity_score
#   else:
#     return 0



def compute_mma_and_shape_scores(gt_seg, samcell_seg, img=None):

  if img == None:
    img = np.ones((gt_seg.shape[0], gt_seg.shape[1]))


  #Compute mma and get segmentation associations
  mma, segmentation_associations = compute_accuracy_max_matching(gt_seg, samcell_seg)

  # shape_score = compute_shape_score(img, gt_seg, samcell_seg, segmentation_associations)

  # return mma, shape_score
  return mma

