import numpy as np
import mahotas
import cv2
import os
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage.morphology import medial_axis, skeletonize

#BASE METHODS

__show_border__ = False

# Read in original image
def get_image(path):
  """
  Get Sample Image
  We get the sample image from the outputs folder and return it as a grayscale image.
  Author(s): ####

  Args:
    img_name (str): name of image folder that contains raw output and original image.

  Returns:
      (np.ndarray): pixel-matrix representation of grayscale image
  """
  image = np.load(path)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return gray_image

def get_segmentations(path):
  """
  Get Sample Segmentation
  We get the sample segmentation from the folder and return it.
  Author(s): ####

  Args:
    image_date (str): name of image and segmentation folder that contains raw output and original image.

  Returns:
      (np.ndarray): pixel-matrix representation of image segmentations
  """
  segmentations = np.load(path)

  return segmentations

"""
This method takes an image name and an optional cell number and returns the corresponding cell mask.
If no cell number is provided, a random cell will be selected.
"""
def get_cell_mask(segmentations, index):

    cell_mask = (segmentations == index).astype(np.uint8)
    if not np.any(cell_mask):
        raise ValueError("There is no cell mask at this index.")
    else:
        return cell_mask


def get_num_cell_segments(segments):
    return len(np.unique(segments))

def area(mask:np.ndarray) -> float:
    """
    Area
    The area of a cell is the number of pixels in its interior.
    We count the number of interior pixels.
    Author(s): ####

    Args:
        mask (np.ndarray): Array of shape (W,H) representing a binary mask of the cell. 1 = cell, 0 = background.

    Returns:
        float: The area of the cell.
    """
    return np.sum(mask, dtype=float)

def area_v2(cropped_cell_mask):
  return np.count_nonzero(cropped_cell_mask)

def bounding_radii(mask:np.ndarray) -> tuple[float, float]:
    """
    Bounding circle radii
    The radii of inscribed and conscribed circles.
    We find the minimum and maximum edge distance from the incenter.
    Author(s): ####

    Args:
        mask (np.ndarray): Array of shape (W,H) representing a binary mask of the cell. 1 = cell, 0 = background.

    Returns:
        tuple: The minimum and maximum edge distance from the incenter.
    """
    in_c = incenter(mask)
    dist = np.linalg.norm(in_c - np.argwhere(get_border_naive(mask)), axis=1)
    return dist.min(), dist.max()

def radius_of_largest_inscribed_circle(mask:np.ndarray):
  dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
  return np.max(dist_map)

def radius_of_smallest_enclosing_circle(mask:np.ndarray):
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnt = max(contours, key=cv2.contourArea)  # largest contour
  (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
  return radius

def radial_profile(mask:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial profile
    The distance from incenter to edge as a function of rotation
    We find the distance from the incenter to each boundary pixel.
    Author(s): ####

    Args:
        mask (np.ndarray): Array of shape (W,H) representing a binary mask of the cell. 1 = cell, 0 = background.

    Returns:
        tuple: The angles and distances from the incenter to each boundary pixel.
    """
    dist = np.argwhere(get_border_naive(mask)) - incenter(mask)
    atan = np.arctan2(dist[:, 0], dist[:, 1])
    return np.sort(atan), np.linalg.norm(dist, axis=1)[np.argsort(atan)]


#########################
    #CELL PARAMETERS
#########################


############################
  #PIXEL INTENSITY METRICS
############################



def get_highlighted_mask(image, mask):
  rgb_image = np.stack((image,) * 3, axis=-1)

  mask_indices = np.where(mask > 0)
  mask_indices_x = mask_indices[0]
  mask_indices_y = mask_indices[1]
  
  for i in range(len(mask_indices_x)):
    rgb_image[mask_indices_x[i]][mask_indices_y[i]] = [255, 0, 0]

  return rgb_image


def get_mask_pixel_intensities(image_crop, cropped_cell_mask):
  """
  Get Mask Pixel Intensities
  We multipy the (grayscaled) original image by the cell mask so that only the pixels of the cell mask (with their original intensities) are preserved.
  All other pixels are set to 0.
  Author(s): ####

  Args:
    image_crop (np.ndarray): array of shape (W,H) representing a grayscale image.
    cropped_cell_mask (np.ndarray): array of shape (W,H) representing a binary mask of the cell. 1 = cell, 0 = background

  Returns:
      (np.ndarray): array of shape (W,H) representing a grayscale image with only the pixels of the cell mask preserved.
  """
  mask_pixel_intensities = image_crop * cropped_cell_mask

  return mask_pixel_intensities



def calculate_gradient_pixel_by_pixel(cropped_image, original_image):
  """
  Calculate Gradient Pixel By Pixel
  Calculates the gradient magnitude and orientation of the cell mask by calculating the gradient magnitude and orienation of each pixel in the mask and averaging over the entire mask.
  Author(s): ####

  Args:
    cropped_image (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    original_image (np.ndarray): array of shape (W,H) representing a grayscale image.

  Returns:
      tuple: The average magnitude and direction of the gradients in the cell mask.
  """
  # inspiration: https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/#:~:text=North:,the%20x%20and%20y%20direction.
  gradient_magnitude = 0
  gradient_orientation = 0

  for i in range(0, len(cropped_image)):
    if np.sum(cropped_image[i]) > 0:
      for j in range(0, len(cropped_image[i])):
        if i > 0 and i < len(cropped_image) - 1 and j > 0 and j < len(cropped_image[i]) - 1 and cropped_image[i][j] > 0 and cropped_image[i+1][j] > 0 and cropped_image[i-1][j] > 0 and cropped_image[i][j+1] > 0 and cropped_image[i][j-1] > 0:
          d_x = int(original_image[i+1][j]) - int(original_image[i-1][j])
          d_y = int(original_image[i][j+1]) - int(original_image[i][j-1])

          magnitude = np.sqrt(d_x**2 + d_y**2)
          orientation = np.arctan2(d_y, d_x) * (180 / np.pi)

          gradient_magnitude += magnitude
          gradient_orientation += orientation

  avg_gradient_magnitude = gradient_magnitude / np.count_nonzero(cropped_image)
  avg_gradient_orientation = gradient_orientation / np.count_nonzero(cropped_image)

  return avg_gradient_magnitude, avg_gradient_orientation


def crop_to_mask(cell_mask, image):
  """
  Crop To Mask
  Crop the masked image to the size of the cell mask.
  Author(s): ####

  Args:
    masked_image (np.ndarray): array of shape (W,H) representing a grayscale image with only the pixels of the cell mask preserved.

  Returns:
      (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
  """
  rows, cols = np.nonzero(cell_mask)
  top, bottom = rows.min(), rows.max()
  left, right = cols.min(), cols.max()

  cropped_cell_mask = cell_mask[top:bottom+1, left:right+1]
  image_crop = image[top:bottom+1, left:right+1]

  return cropped_cell_mask, image_crop, (top, bottom, left, right)


def calculate_gradient_layer_by_layer(cropped_image):
  """
  Calculate Gradient Layer by Layer
  Calculates the gradient magnitude and orientation of the cell mask.
  First computes the average pixel intensity of each layer of the cell mask.
  Calculates the absolute difference between the average pixel intensity in one layer to the next (moving inwards) and sums over each layer of the mask.
  Author(s): ####

  Args:
    cropped_image (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      tuple(np.ndarray, float): An array of the average pixel intensities from each layer, and the summation of change in average pixel intensity from layer to layer (the gradient metric).
  """
  cont = True
  cropped_image_last = cropped_image.copy()
  cropped_image_next = cropped_image.copy()
  layer_avgs = []
  while cont:
    removed_layer = []
    possible_points = np.argwhere(cropped_image_last > 0)
    for point in possible_points:
      if point[0] > 0 and point[0] < len(cropped_image_last) - 1 and point[1] > 0 and point[1] < len(cropped_image_last[point[0]]) - 1 and cropped_image_last[point[0] + 1][point[1]] > 0 and cropped_image_last[point[0] - 1][point[1]] > 0 and cropped_image_last[point[0]][point[1] + 1] > 0 and cropped_image_last[point[0]][point[1] - 1] > 0:
        continue
      else:
        cropped_image_next[point[0]][point[1]] = 0
        removed_layer.append(cropped_image[point[0]][point[1]])

    layer_avgs.append(np.mean(removed_layer))
    cropped_image_last = cropped_image_next.copy()
    if np.sum(cropped_image_last) == 0:
      cont = False

  gradient_metric = 0
  for i in range(1, len(layer_avgs)):
    gradient_metric += np.abs(layer_avgs[i] - layer_avgs[i-1])

  #gradient_metric /= len(layer_avgs)

  return layer_avgs, gradient_metric


def calculate_gradient_layer_by_layer_v2(cropped_image, cropped_cell_mask, border_coords):
  """
  Calculate Gradient Layer by Layer
  Calculates the gradient magnitude and orientation of the cell mask.
  First computes the average pixel intensity of each layer of the cell mask.
  Calculates the absolute difference between the average pixel intensity in one layer to the next (moving inwards) and sums over each layer of the mask.
  Author(s): ####

  Args:
    cropped_image (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      tuple(np.ndarray, float): An array of the average pixel intensities from each layer, and the summation of change in average pixel intensity from layer to layer (the gradient metric).
  """
  cont = True
  border = []
  for point in border_coords:
    border.append(cropped_image[point[0]][point[1]])
    cropped_cell_mask[point[0]][point[1]] = 0
  layer_1_mean_intensity = np.mean(border)
  layer_avgs = [layer_1_mean_intensity]
  while np.sum(cropped_cell_mask) > 0:
    #get border coords of current mask
    border_coords = get_border_coords(cropped_cell_mask)
    border = []
    for point in border_coords:
      border.append(cropped_image[point[0]][point[1]])
      cropped_cell_mask[point[0]][point[1]] = 0

    #calculate mean intensity of border
    removed_layer_mean_intensity = np.mean(border)
    layer_avgs.append(removed_layer_mean_intensity)

  gradient_metric = 0
  for i in range(1, len(layer_avgs)):
    gradient_metric += np.abs(layer_avgs[i] - layer_avgs[i-1])

  #gradient_metric /= len(layer_avgs)

  return layer_avgs, gradient_metric


def display_layer_gradient(cropped_image):
  """
  Display Layer Gradient
  Computes the average pixel intensity of each layer of the cell mask and displays the mask with each layer colored by its average pixel intensity.
  Author(s): ####

  Args:
    cropped_image (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      N/A
  Displays:
      Cropped cell mask with each layer colored by its average pixel intensity.
  """
  cont = True
  layer_avgs = []
  layer_coords = []
  while cont:
    new_mask = []
    removed_layer = []
    curr_layer_coords = []
    for i in range(len(cropped_image)):
      new_mask_row = []
      for j in range(len(cropped_image[i])):
        if cropped_image[i][j] > 0 and i > 0 and i < len(cropped_image) - 1 and j > 0 and j < len(cropped_image[i]) -1 and cropped_image[i+1][j] > 0 and cropped_image[i-1][j] > 0 and cropped_image[i][j+1] > 0 and cropped_image[i][j-1] > 0:
            new_mask_row.append(cropped_image[i][j])
        else:
          new_mask_row.append(0)
          if cropped_image[i][j] > 0:
            removed_layer.append(cropped_image[i][j])
            curr_layer_coords.append((i, j))
      new_mask.append(new_mask_row)

    layer_avgs.append(np.mean(removed_layer))
    layer_coords.append(curr_layer_coords)
    if np.sum(new_mask) == 0:
      cont = False
    else:
      cropped_image = np.array(new_mask)

  new_img = cropped_image.copy()
  for i in range(len(cropped_image)):
    for j in range(len(cropped_image[i])):
      for k in range(len(layer_coords)):
        if (i, j) in layer_coords[k]:
          new_img[i][j] = layer_avgs[k]

  plt.imshow(new_img, cmap='gray')
  plt.show()


  #gradient_metric /= len(layer_avgs)


def calculate_pixel_intensity_metrics(cropped_image, display_histogram=False):
  """
  Calculate Pixel Intensity Metrics
  Computes the mean intensity, standard deviation of intensity, max intensity, min intensity, and intensity range of the cell mask.
  Author(s): ####

  Args:
    cropped_image (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      tuple: The mean intensity, standard deviation of intensity, max intensity, min intensity, and intensity range of the cell mask.
  Displays:
      Histogram of pixel intensities in the cell mask.
  """
  # source: https://static-content.springer.com/esm/art%3A10.1186%2Fgb-2006-7-10-r100/MediaObjects/13059_2006_1368_MOESM4_ESM.pdf
    # for metric idea
  #display_histogram = True
  mean_intensity = np.mean(cropped_image, where=cropped_image > 0, dtype=float)
  std_intensity = np.std(cropped_image, where=cropped_image > 0, dtype=float)
  max_intensity = float(np.max(cropped_image, where=cropped_image > 0, initial=0))
  min_intensity = float(np.min(cropped_image, where=cropped_image > 0, initial=255))
  intensity_range = max_intensity - min_intensity

  if display_histogram:
    flattened_cropped_image = cropped_image.flatten()
    # plt.hist(flattened_cropped_image, bins='auto', range=(min_intensity, max_intensity))
    # plt.title('Histogram of Pixel Intensities')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')
    #plt.show()

  # haralick measures "texture" of the image (returns 4x13 matrix describing texture)
    # sources:
      #https://cvexplained.wordpress.com/2020/07/22/10-6-haralick-texture/
      #https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.23984#support-information-section
  # features = mahotas.features.haralick(cropped_image).mean(axis=0)
  # features = mahotas.features.haralick(cropped_image)
  # print(features)

  return mean_intensity, std_intensity, max_intensity, min_intensity, intensity_range


def count_pixel_neighbors(mask, i, j):
  """
  Count Pixel Neighbors
  Counts the number of pixels bordering a pixel (left, right, above, below)
  Author(s): ####

  Args:
    mask (np.ndarray): Array of shape (W,H) representing a binary mask of the cell. 1 = cell, 0 = background.

  Returns:
      int: The number of pixels bordering a pixel (left, right, above, below).
  """
  neighbors = 0

  if i > 0 and mask[i-1][j] > 0:
    neighbors += 1
  if i < len(mask) - 1 and mask[i+1][j] > 0:
    neighbors += 1
  if j > 0 and mask[i][j-1] > 0:
    neighbors += 1
  if j < len(mask[i]) - 1 and mask[i][j+1] > 0:
    neighbors += 1

  return neighbors

def perimeter_by_border(mask):
  """
  Perimeter By Border
  Calculates perimeter of the mask by counting the number of edges in the mask.
  For each pixel in the border of the mask, calculate 4 - its number of neighbors, sum over each pixel in the border.
  Author(s): ####

  Args:
    mask (np.ndarray): Array of shape (W,H) representing a binary mask of the cell. 1 = cell, 0 = background.

  Returns:
      int: The perimeter of the mask represented by the number of edges in the mask.
  """
  perimeter = 0
  #curr_layer_coords = []
  for i in range(len(mask)):
    if np.sum(mask[i]) == 0:
      continue
    for j in range(len(mask[i])):
      if mask[i][j] > 0:
        if i > 0 and i < len(mask) - 1 and j > 0 and j < len(mask[i]) -1 and mask[i+1][j] > 0 and mask[i-1][j] > 0 and mask[i][j+1] > 0 and mask[i][j-1] > 0:
            continue
        else:
          perimeter += (4 - count_pixel_neighbors(mask, i, j))
          #curr_layer_coords.append((i, j))

  return perimeter





########################
  #SURROUNDING CELLS
########################

def get_surrounding_cells(cell_mask, anns, crop_coords, cell_index):
  top = 0 if crop_coords[0] == 0 else crop_coords[0] - 1
  bottom = len(cell_mask) - 1 if crop_coords[1] == len(cell_mask) - 1 else crop_coords[1] + 1
  left = 0 if crop_coords[2] == 0 else crop_coords[2] - 1
  right = len(cell_mask[0]) - 1 if crop_coords[3] == len(cell_mask[0]) - 1 else crop_coords[3] + 1

  cell_mask = cell_mask[top:bottom+1, left:right+1]
  anns = anns[top:bottom+1, left:right+1]

  dilated_mask = cv2.dilate(cell_mask, np.ones((3,3), np.uint8), iterations=1)
  touching_masks = anns * dilated_mask
  touching_masks = np.unique(touching_masks)
  touching_masks = np.delete(touching_masks, np.where(touching_masks == 0))
  touching_masks = np.delete(touching_masks, np.where(touching_masks == cell_index))

  return touching_masks, len(touching_masks)

###########################
  #MAJOR AND MINOR AXES
###########################

def get_major_minor_axis_and_theta(mask, plot=False):
  mask = (mask > 0).astype(np.uint8)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  ellipse = cv2.fitEllipse(contours[0])
  (center, axes, angle) = ellipse
  cx, cy = center
  minor, major = axes
  theta = np.deg2rad(angle)

  if plot:
    # Generate ellipse points
    t = np.linspace(0, 2*np.pi, 100)
    ellipse_x = (major/2) * np.cos(t)
    ellipse_y = (minor/2) * np.sin(t)

    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    rotated = np.dot(R, np.array([ellipse_x, ellipse_y]))

    # Shift to center
    ellipse_x_rot = rotated[0] + cx
    ellipse_y_rot = rotated[1] + cy

    # Plot
    plt.imshow(mask, cmap='gray')
    plt.plot(ellipse_x_rot, ellipse_y_rot, 'r', linewidth=2)
    plt.scatter([cx], [cy], c='blue', s=40)  # center point
    plt.show()

  return major, minor, theta



#####################
    #CONVEX HULL
#####################
def get_border_pixels_old(cropped_cell_mask):
  """
  Get Border Pixels
  Finds and returns the border pixels of the segmentation.
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      NDArray: The border pixels of the segmentation
  """
  border_points = []
  for i in range(len(cropped_cell_mask)):
    for j in range(len(cropped_cell_mask[i])):
      if cropped_cell_mask[i][j] > 0:
        if i > 0 and i < len(cropped_cell_mask) - 1 and j > 0 and j < len(cropped_cell_mask[i]) -1 and cropped_cell_mask[i+1][j] > 0 and cropped_cell_mask[i-1][j] > 0 and cropped_cell_mask[i][j+1] > 0 and cropped_cell_mask[i][j-1] > 0:
          continue
        else:
          border_points.append((i, j))

  return border_points

def get_border_coords(mask):
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    coords = []
    for contour in contours:
        coords.extend([(int(pt[0][1]), int(pt[0][0])) for pt in contour])  # (y, x) format
    return coords




def plot_border(cropped_cell_mask):
  """
  Plot Border
  Plots border of the segmentation.
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      N/A

  Displays:
      The border of the segmentation
  """
  border_points = get_border_coords(cropped_cell_mask)
  new_img = np.zeros((cropped_cell_mask.shape[0], cropped_cell_mask.shape[1]))
  for point in border_points:
    new_img[point[0]][point[1]] = 255

  #plt.imshow(new_img, cmap='gray')
  #plt.show()

def cross_multiply(p1, p2, p3):
  """
  Cross Multiply
  Cross multiplies three points and returns the result
  Author(s): ####

  Args:
    p1: first point
    p2: second point
    p3: third point

  Returns:
      float: The cross of the first, second, and third point
  """
  return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


def get_convex_hull(cropped_cell_mask, border_points=None):
  """
  Get Convex Hull
  Returns the convex hull of the segmentation using Andrew's monotone chain convex hull algorithm
  Source: https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
  Author(s): ####

  Args:
      cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      NDArray: The vertices and edges of the convex hull
  """
  if border_points == None:
    border_points = sorted(get_border_coords(cropped_cell_mask)) # sorts by x-coords, sorts by y-coord if tie
  else:
    border_points = sorted(border_points)

  lower_hull = []
  for p in border_points:
      while len(lower_hull) >= 2 and cross_multiply(lower_hull[-2], lower_hull[-1], p) <= 0:
          lower_hull.pop()
      lower_hull.append(p)

  upper_hull = []
  for p in reversed(border_points):
      while len(upper_hull) >= 2 and cross_multiply(upper_hull[-2], upper_hull[-1], p) <= 0:
          upper_hull.pop()
      upper_hull.append(p)


  full_hull_vertices = lower_hull[:-1] + upper_hull[:-1]

  # new_img = np.zeros_like(cropped_cell_mask)
  # for point in border_points:
  #   new_img[point[0]][point[1]] = 255
  # for point in full_hull_vertices:
  #   new_img[point[0]][point[1]] = 100

  full_hull_edges = []
  for i in range(len(full_hull_vertices)):
    x1, y1 = full_hull_vertices[i]
    x2, y2 = full_hull_vertices[(i+1) % len(full_hull_vertices)]
    full_hull_edges.append([(x1, y1), (x2, y2)])
    #plt.plot([y1, y2], [x1, x2], color='red')

  # plt.imshow(new_img, cmap='gray')
  # plt.show()

  return full_hull_vertices, full_hull_edges


def get_convex_hull_area(cropped_cell_mask, convex_hull_edges=None):
  """
  Get Convex Hull Area
  Gauss's Shoelace Formula
  Source: https://blogs.sas.com/content/iml/2022/11/02/area-perimeter-convex-hull.html
  Source 2: https://en.wikipedia.org/wiki/Shoelace_formula
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      float: the area of the convex hull of the segmentation
  """
  if convex_hull_edges == None:
    convex_hull_vertices, convex_hull_edges = get_convex_hull(cropped_cell_mask)
  area = 0
  for edge in convex_hull_edges:
    area += edge[0][0] * edge[1][1] - edge[1][0] * edge[0][1]
  area = np.abs(area) / 2
  return area


def get_convex_hull_perimeter(cropped_cell_mask, convex_hull_edges=None):
  """
  Get Convex Hull Perimeter
  Sums the lengths of the edges of the convex hull to get its perimeter
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      float: the perimeter of the convex hull of the segmentation
  """
  if convex_hull_edges == None:
    convex_hull_vertices, convex_hull_edges = get_convex_hull(cropped_cell_mask)
  perimeter = 0
  for edge in convex_hull_edges:
    perimeter += np.sqrt(np.power(edge[1][0] - edge[0][0], 2) + np.power(edge[1][1] - edge[0][1], 2))
  return perimeter



def get_convex_hull_v2(border_points):
  hull = cv2.convexHull(np.array(border_points))
  area = cv2.contourArea(hull)
  perimeter = cv2.arcLength(hull, True)
  return hull, area, perimeter


##########################
  #MINIMUM BOUNDING BOX
##########################

# Rotating Calipers Algorithms
# source: https://www.geometrictools.com/Documentation/MinimumAreaRectangle.pdf
def convert_hull_vertices_to_new_coordinate_plane(hull_vertices, angle):
  """
  Convert Hull Vertices to New Coordinate Plane
  Converts the hull vertices to a new coordinate plane based on the angle of the convex hull edge
  Author(s): ####

  Args:
      hull_vertices (np.ndarray): the vertices of the convex hull
      angle (float): the angle of the convex hull edge

  Returns:
      NDArray: The vertices of the convex hull in the new coordinate plane
  """
  converted_hull_vertices = []
  cos_angle = np.cos(angle)
  sin_angle = np.sin(angle)
  for vertex in hull_vertices:
    converted_vertice_x = (cos_angle * vertex[0]) - (sin_angle * vertex[1])
    converted_vertice_y = (sin_angle * vertex[0]) + (cos_angle * vertex[1])
    converted_hull_vertices.append((converted_vertice_x, converted_vertice_y))

  return converted_hull_vertices

def convert_to_original_plane(converted_vertice, angle):
  """
  Convert to Original Plane
  Converts the hull vertices back to the originaal coordinate plane
  Author(s): ####

  Args:
      converted_vertices (np.ndarray): the vertices of the convex hull converted to the new plane
      angle (float): the angle of the convex hull edge

  Returns:
      NDArray: The vertices of the convex hull in the original coordinate plane
  """
  original_vertice_x = np.cos(-angle) * (converted_vertice[0]) - np.sin(-angle) * (converted_vertice[1])
  original_vertice_y = np.sin(-angle) * (converted_vertice[0]) + np.cos(-angle) * (converted_vertice[1])
  return (original_vertice_x, original_vertice_y)

def find_bounding_box(converted_hull_vertices):
  """
  Find Bounding Box
  Finds the Bounding Box of converted hull vertices
  Author(s): ####

  Args:
      converted_hull_vertices (np.ndarray): the vertices of the convex hull converted to the new plane

  Returns:
      NDArray: The vertices of the bounding box of the converted hull vertices
  """
  max_x = np.max(np.array(converted_hull_vertices)[:, 0])
  min_x = np.min(np.array(converted_hull_vertices)[:, 0])
  max_y = np.max(np.array(converted_hull_vertices)[:, 1])
  min_y = np.min(np.array(converted_hull_vertices)[:, 1])

  top_left = (min_x, min_y)
  top_right = (min_x, max_y)
  bottom_left = (max_x, min_y)
  bottom_right = (max_x, max_y)

  return (top_left, top_right, bottom_left, bottom_right)

def get_minimum_bounding_box(cropped_cell_mask):
  """
  Get Minimum Bounding Box
  Finds the Bounding Box with the minimum area by iterating through different edges of the convex hull as starting edges for the bounding box
  Rotating Calipers Algorithm
  Source: https://www.geometrictools.com/Documentation/MinimumAreaRectangle.pdf
  Author(s): ####

  Args:
      cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      NDArray: The vertices of the minimum area bounding box in vertices of the original plane
  """
  padded_image = np.pad(cropped_cell_mask, 40, mode='constant', constant_values=0)
  convex_hull_vertices, convex_hull_edges = get_convex_hull(padded_image)

  minimum_area_bounding_box = 1000000
  minimum_area_bounding_box_vertices = []
  minimum_perimeter_bounding_box = 1000000
  minimum_perimeter_bounding_box_vertices = []
  for edge in convex_hull_edges:
    initial_axis_dir = np.array((edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]) / np.sqrt(np.power(edge[1][0] - edge[0][0], 2) + np.power(edge[1][1] - edge[0][1], 2)))
    angle = np.arctan2(initial_axis_dir[1], initial_axis_dir[0])
    #print(edge)
    #print(initial_axis_dir)
    #print(angle * (180/np.pi))
    #perp_axis_dir = np.array((-initial_axis_dir[1], initial_axis_dir[0]))
    converted_hull_vertices = convert_hull_vertices_to_new_coordinate_plane(convex_hull_vertices, angle)
    #print(converted_hull_vertices)
    top_left_converted, top_right_converted, bottom_left_converted, bottom_right_converted = find_bounding_box(converted_hull_vertices)
    top_left = np.round(convert_to_original_plane(top_left_converted, angle))
    top_right = np.round(convert_to_original_plane(top_right_converted, angle))
    bottom_left = np.round(convert_to_original_plane(bottom_left_converted, angle))
    bottom_right = np.round(convert_to_original_plane(bottom_right_converted, angle))

    top_length = np.sqrt(np.power(top_right[0] - top_left[0], 2) + np.power(top_right[1] - top_left[1], 2))
    side_length = np.sqrt(np.power(bottom_right[0] - top_right[0], 2) + np.power(bottom_right[1] - top_right[1], 2))
    area = top_length * side_length
    perimeter = 2 * (top_length + side_length)
    if area < minimum_area_bounding_box:
      minimum_area_bounding_box = area
      minimum_area_bounding_box_vertices = [top_left, top_right, bottom_left, bottom_right]
    if perimeter < minimum_perimeter_bounding_box:
      minimum_perimeter_bounding_box = perimeter
      minimum_perimeter_bounding_box_vertices = [top_left, top_right, bottom_left, bottom_right]

    # plt.plot([top_left[1], top_right[1]], [top_left[0], top_right[0]], color='red')
    # plt.plot([top_left[1], bottom_left[1]], [top_left[0], bottom_left[0]], color='red')
    # plt.plot([top_right[1], bottom_right[1]], [top_right[0], bottom_right[0]], color='red')
    # plt.plot([bottom_left[1], bottom_right[1]], [bottom_left[0], bottom_right[0]], color='red')

    # plt.imshow(padded_image, cmap='gray')
    # plt.show()


  # plt.title('Minimum Area Bounding Box')
  # plt.plot([minimum_area_bounding_box_vertices[0][1], minimum_area_bounding_box_vertices[1][1]], [minimum_area_bounding_box_vertices[0][0], minimum_area_bounding_box_vertices[1][0]], color='red')
  # plt.plot([minimum_area_bounding_box_vertices[0][1], minimum_area_bounding_box_vertices[2][1]], [minimum_area_bounding_box_vertices[0][0], minimum_area_bounding_box_vertices[2][0]], color='red')
  # plt.plot([minimum_area_bounding_box_vertices[1][1], minimum_area_bounding_box_vertices[3][1]], [minimum_area_bounding_box_vertices[1][0], minimum_area_bounding_box_vertices[3][0]], color='red')
  # plt.plot([minimum_area_bounding_box_vertices[2][1], minimum_area_bounding_box_vertices[3][1]], [minimum_area_bounding_box_vertices[2][0], minimum_area_bounding_box_vertices[3][0]], color='red')

  # plt.imshow(padded_image, cmap='gray')
  # plt.show()

  # plt.title('Minimum Perimeter Bounding Box')
  # plt.plot([minimum_perimeter_bounding_box_vertices[0][1], minimum_perimeter_bounding_box_vertices[1][1]], [minimum_perimeter_bounding_box_vertices[0][0], minimum_perimeter_bounding_box_vertices[1][0]], color='red')
  # plt.plot([minimum_perimeter_bounding_box_vertices[0][1], minimum_perimeter_bounding_box_vertices[2][1]], [minimum_perimeter_bounding_box_vertices[0][0], minimum_perimeter_bounding_box_vertices[2][0]], color='red')
  # plt.plot([minimum_perimeter_bounding_box_vertices[1][1], minimum_perimeter_bounding_box_vertices[3][1]], [minimum_perimeter_bounding_box_vertices[1][0], minimum_perimeter_bounding_box_vertices[3][0]], color='red')
  # plt.plot([minimum_perimeter_bounding_box_vertices[2][1], minimum_perimeter_bounding_box_vertices[3][1]], [minimum_perimeter_bounding_box_vertices[2][0], minimum_perimeter_bounding_box_vertices[3][0]], color='red')

  # plt.imshow(padded_image, cmap='gray')
  # plt.show()

  return minimum_area_bounding_box_vertices#, minimum_perimeter_bounding_box_vertices


def get_minimum_bounding_box_v2(convex_hull, cropped_cell_mask):
  # Rotating calipers
  rect = cv2.minAreaRect(convex_hull) # ((cx, cy), (w, h), angle)
  ((cx, cy), (w, h), angle) = rect
  box_vertices = cv2.boxPoints(rect) # bounding box vertices

  return box_vertices, w, h




#########################
  #NEW PERIMETER METRIC
#########################

def get_perimeter_new(cropped_cell_mask, border_points = None):
  """
  Get Perimeter New
  8  1  2
  7  x  3
  6  5  4
  x = curr point
  numbers are order in which surrouding pixels are looked at
  Mimics the clockwise traversal algorithm, summing the perimiter by adding 1 if the next point
  is straight from the current point, or sqrt(2) if the next point is diagonal
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      float: the perimeter of the cell segmentation
  """
  if border_points == None:
    border_points = get_border_coords(cropped_cell_mask)

  leftmost_point_index = np.argmin(np.array(border_points)[:, 1])
  leftmost_point = border_points[leftmost_point_index]
  traversal = [leftmost_point]

  perimeter = 0
  current_point = leftmost_point
  while True:
    next_points = [(current_point[0] - 1, current_point[1]),
                   (current_point[0] - 1, current_point[1] + 1),
                   (current_point[0], current_point[1] + 1),
                   (current_point[0] + 1, current_point[1] + 1),
                   (current_point[0] + 1, current_point[1]),
                   (current_point[0] + 1, current_point[1] - 1),
                   (current_point[0], current_point[1] - 1),
                   (current_point[0] - 1, current_point[1] - 1)]
    straight_points = [(current_point[0] - 1, current_point[1]),
                   (current_point[0], current_point[1] + 1),
                   (current_point[0] + 1, current_point[1]),
                   (current_point[0], current_point[1] - 1)]
    diagonal_points = [(current_point[0] - 1, current_point[1] + 1),
                    (current_point[0] + 1, current_point[1] + 1),
                   (current_point[0] + 1, current_point[1] - 1),
                   (current_point[0] - 1, current_point[1] - 1)]
    found = False
    for next_point in next_points:
      if found == False:
        if next_point in border_points:
          if next_point not in traversal:
            traversal.append(next_point)
            current_point = next_point
            if next_point in straight_points:
              perimeter += 1
            elif next_point in diagonal_points:
              perimeter += np.sqrt(2)
            found = True
    if found == False:
      if len(traversal) == len(border_points):
        break
      else:
        end = False
        found_2 = False
        back = 1
        while found_2 == False:
          if back > 20:
            end = True
            break
          curr_point = traversal[-back]
          next_points = [(curr_point[0] - 1, curr_point[1]),
                   (curr_point[0] - 1, curr_point[1] + 1),
                   (curr_point[0], curr_point[1] + 1),
                   (curr_point[0] + 1, curr_point[1] + 1),
                   (curr_point[0] + 1, curr_point[1]),
                   (curr_point[0] + 1, curr_point[1] - 1),
                   (curr_point[0], curr_point[1] - 1),
                   (curr_point[0] - 1, curr_point[1] - 1)]
          straight_points = [(current_point[0] - 1, current_point[1]),
                        (current_point[0], current_point[1] + 1),
                        (current_point[0] + 1, current_point[1]),
                        (current_point[0], current_point[1] - 1)]
          diagonal_points = [(current_point[0] - 1, current_point[1] + 1),
                          (current_point[0] + 1, current_point[1] + 1),
                        (current_point[0] + 1, current_point[1] - 1),
                        (current_point[0] - 1, current_point[1] - 1)]
          found = False
          for next_point in next_points:
            if found == False:
              if next_point in border_points:
                if next_point not in traversal:
                  traversal.append(next_point)
                  current_point = next_point
                  if next_point in straight_points:
                    perimeter += 1
                  elif next_point in diagonal_points:
                    perimeter += np.sqrt(2)
                  found = True
                  found_2 = True
          if found == False:
            back += 1
        if end == True:
          break

  return perimeter


def get_perimeter_new_v2(cropped_cell_mask, sorted_border_points=None):
  """
  Get Perimeter New
  8  1  2
  7  x  3
  6  5  4
  x = curr point
  numbers are order in which surrouding pixels are looked at
  Mimics the clockwise traversal algorithm, summing the perimiter by adding 1 if the next point
  is straight from the current point, or sqrt(2) if the next point is diagonal
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      float: the perimeter of the cell segmentation
  """
  if sorted_border_points is None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

  perimeter = 0
  last_point = sorted_border_points[0]
  for point in sorted_border_points[1:]:
    if point[0] == last_point[0] or point[1] == last_point[1]:
      perimeter += 1
    else:
      perimeter += np.sqrt(2)

    last_point = point

  #account for last to first point
  first_point = sorted_border_points[0]
  last_point = sorted_border_points[-1]
  if last_point[0] == first_point[0] or last_point[1] == first_point[1]:
    perimeter += 1
  else:
    perimeter += np.sqrt(2)

  return perimeter

def get_perimeter_v3(sorted_border_points=None, cropped_cell_mask=None):
  if sorted_border_points is None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

  pts = np.asarray(sorted_border_points, dtype=np.int32)
  # append first point to end for wraparound
  pts = np.vstack([pts, pts[0]])

  # differences between consecutive points
  diffs = np.abs(np.diff(pts, axis=0))

  # straight moves = one coordinate changes
  straight_moves = np.any(diffs == 0, axis=1)
  diag_moves = ~straight_moves

  perimeter = np.count_nonzero(straight_moves) * 1.0 + np.count_nonzero(diag_moves) * np.sqrt(2)
  return perimeter



###############################
  #CLOCKWISE BORDER TRAVERSAL
###############################

def get_clockwise_border_traversal(cropped_cell_mask, border_points=None, plot=False):
  """
  Get Clockwise Border Traversal
  8  1  2
  7  x  3
  6  5  4
  x = curr point
  numbers are order in which surrouding pixels are looked at
  Starting at the leftmost point, find the next point by searching for it in order show above
  Continue until all points have been visited (may ignore some points that should not be a part of the border)
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      NDArray: the clockwise traversal of the border points
  """
  if border_points is None:
    border_points = get_border_coords(cropped_cell_mask)

  leftmost_point_index = np.argmin(np.array(border_points)[:, 1])
  leftmost_point = border_points[leftmost_point_index]
  traversal = [leftmost_point]

  #create sets for efficiency
  border_set = set(border_points)
  traversal_set = set(traversal)

  current_point = leftmost_point
  while True:
    next_points = [(current_point[0] - 1, current_point[1]),
                   (current_point[0] - 1, current_point[1] + 1),
                   (current_point[0], current_point[1] + 1),
                   (current_point[0] + 1, current_point[1] + 1),
                   (current_point[0] + 1, current_point[1]),
                   (current_point[0] + 1, current_point[1] - 1),
                   (current_point[0], current_point[1] - 1),
                   (current_point[0] - 1, current_point[1] - 1)]
    found = False
    for next_point in next_points:
      if found == False:
        if next_point in border_set:
          if next_point not in traversal_set:
            traversal.append(next_point)
            traversal_set.add(next_point)
            current_point = next_point
            found = True
    if found == False:
      if len(traversal) == len(border_points):
        break
      else:
        end = False
        found_2 = False
        back = 1
        while found_2 == False:
          if back > len(traversal_set):
            end = True
            break
          curr_point = traversal[-back]
          next_points = [(curr_point[0] - 1, curr_point[1]),
                   (curr_point[0] - 1, curr_point[1] + 1),
                   (curr_point[0], curr_point[1] + 1),
                   (curr_point[0] + 1, curr_point[1] + 1),
                   (curr_point[0] + 1, curr_point[1]),
                   (curr_point[0] + 1, curr_point[1] - 1),
                   (curr_point[0], curr_point[1] - 1),
                   (curr_point[0] - 1, curr_point[1] - 1)]
          found = False
          for next_point in next_points:
            if found == False:
              if next_point in border_set:
                if next_point not in traversal_set:
                  traversal.append(next_point)
                  traversal_set.add(next_point)
                  current_point = next_point
                  found = True
                  found_2 = True
          if found == False:
            back += 1
        if end == True:
          break


  # if plot == True:
  #   fig, ax = plt.subplots(1, 2)
  #   ax[0].imshow(cropped_cell_mask, cmap='gray')

  #   new_plot = np.zeros_like(cropped_cell_mask)
  #   curr_color = 255.0
  #   for point in traversal:
  #     new_plot[point[0]][point[1]] = int(curr_color)
  #     curr_color -= (255 / len(traversal))
  #   ax[1].imshow(new_plot, cmap='gray')
  #   plt.show()
  return traversal





####################
  #CELL CENTER
####################


def find_center(cropped_cell_mask):
  """
  Find Center
  Finds the visual center of the image by stripping away the layers of the cell mask until reaching the center pixels.
  If multiple pixels remain in final layer, calculate average of their coords and find the closest pixel to that average.
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      tuple: Row, column coordinates of the center of the cell mask.
  """
  cont = True
  original_cropped_cell_mask = cropped_cell_mask.copy()
  while cont:
    new_mask = []
    for i in range(len(cropped_cell_mask)):
      new_mask_row = []
      for j in range(len(cropped_cell_mask[i])):
        if cropped_cell_mask[i][j] > 0 and i > 0 and i < len(cropped_cell_mask) - 1 and j > 0 and j < len(cropped_cell_mask[i]) -1 and cropped_cell_mask[i+1][j] > 0 and cropped_cell_mask[i-1][j] > 0 and cropped_cell_mask[i][j+1] > 0 and cropped_cell_mask[i][j-1] > 0:
            new_mask_row.append(cropped_cell_mask[i][j])
        else:
          new_mask_row.append(0)
      new_mask.append(new_mask_row)

    if np.sum(new_mask) == 0:
      cont = False
    else:
      cropped_cell_mask = np.array(new_mask)

  if np.count_nonzero(cropped_cell_mask) > 1:
    x_tot = 0
    y_tot = 0
    for i in range(len(cropped_cell_mask)):
      if np.sum(cropped_cell_mask[i]) > 0:
        for j in range(len(cropped_cell_mask[i])):
          if cropped_cell_mask[i][j] > 0:
            x_tot += i
            y_tot += j

    x_center = x_tot / np.count_nonzero(cropped_cell_mask)
    y_center = y_tot / np.count_nonzero(cropped_cell_mask)

    closest_x = 0
    closest_y = 0
    closest_dist = 1500
    for i in range(len(original_cropped_cell_mask)):
      if np.sum(original_cropped_cell_mask[i]) > 0:
        for j in range(len(original_cropped_cell_mask[i])):
          if original_cropped_cell_mask[i][j] > 0:
            x_dist = i - x_center
            y_dist = j - y_center
            dist = np.sqrt(x_dist**2 + y_dist**2)
            if dist < closest_dist:
              closest_dist = dist
              closest_x = i
              closest_y = j

    cropped_cell_mask = np.zeros(cropped_cell_mask.shape)
    cropped_cell_mask[closest_x][closest_y] = 255

  #plt.imshow(cropped_cell_mask, cmap='gray')
  #plt.show()

  center_coords = np.where(cropped_cell_mask > 0)

  return (center_coords[0][0], center_coords[1][0])


def find_center_new(cropped_cell_mask):
  """
  Find Center
  Finds the visual center of the image by finding the pixel that has the minimal sum of the different between the pixels below/above it and left/right of it.
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      tuple: Row, column coordinates of the center of the cell mask.
  """
  closest_center_metric = 100000
  closest_center_coords = (0, 0)
  for i in range(len(cropped_cell_mask)):
    for j in range(len(cropped_cell_mask[i])):
      if cropped_cell_mask[i][j] > 0:
        num_above = np.count_nonzero(cropped_cell_mask[0:i, :])
        num_below = np.count_nonzero(cropped_cell_mask[i:len(cropped_cell_mask), :])
        num_left = np.count_nonzero(cropped_cell_mask[:, 0:j])
        num_right = np.count_nonzero(cropped_cell_mask[:, j:len(cropped_cell_mask[i])])

        center_metric = np.abs(num_above - num_below) + np.abs(num_left - num_right)
        if center_metric < closest_center_metric:
          closest_center_metric = center_metric
          closest_center_coords = (i, j)

  return closest_center_coords


def find_best_center(cropped_cell_mask, sorted_border_points=None):
  """
  Find Best Center
  Calculates the radial distances of the sorted border points for each possible center
  Finds the center with the least variance in radial distances
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      tuple: the coordinates of the center with the least variance in radial distances
  """
  minimum_variance_radial_distance_center = 100000000000
  minimum_variance_center_coords = (0, 0)

  if sorted_border_points == None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

  test_points = np.argwhere(cropped_cell_mask > 0)

  squared_x_distances = []
  squared_y_distances = []
  for i in range(np.min(test_points[:, 0]), np.max(test_points[:, 0]) + 1):
    squared_x_distances.append(np.power(np.array(sorted_border_points)[:, 0] - i, 2))
  for j in range(np.min(test_points[:, 1]), np.max(test_points[:, 1]) + 1):
    squared_y_distances.append(np.power(np.array(sorted_border_points)[:, 1] - j, 2))

  for point in test_points:
    radial_distances = np.sqrt(squared_x_distances[point[0]] + squared_y_distances[point[1]])

    variance = np.var(radial_distances)

    if variance < minimum_variance_radial_distance_center:
      minimum_variance_radial_distance_center = variance
      minimum_variance_center_coords = (point[0], point[1])

  return minimum_variance_center_coords


def find_best_center_new(cropped_cell_mask):
    prev_mask = deepcopy(cropped_cell_mask)
    
    curr_mask = cv2.erode(cropped_cell_mask, np.ones((3,3), np.uint8), iterations=1)
    while np.sum(curr_mask) > 0:
        prev_mask = deepcopy(curr_mask)
        curr_mask = cv2.erode(curr_mask, np.ones((3,3), np.uint8), iterations=1)

    remaining_pixels = np.where(prev_mask == 1)
    if np.sum(prev_mask) > 1:
        mean_x = round(np.mean(remaining_pixels[0]) + 1e-4)
        mean_y = round(np.mean(remaining_pixels[1]) + 1e-4)
        mean_coords = (mean_x, mean_y)
        return mean_coords
    else:
        return (remaining_pixels[0], remaining_pixels[1])



def find_best_center_new_v2(cropped_cell_mask, sorted_border_points=None):
    minimum_variance_radial_distance_center = 100000000000
    minimum_variance_center_coords = (0, 0)

    if sorted_border_points == None:
        sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

    prev_mask = deepcopy(cropped_cell_mask)
    
    curr_mask = cv2.erode(cropped_cell_mask, np.ones((3,3), np.uint8), iterations=1)
    while np.sum(curr_mask) > 40:
        prev_mask = deepcopy(curr_mask)
        curr_mask = cv2.erode(curr_mask, np.ones((3,3), np.uint8), iterations=1)

    test_points = np.argwhere(prev_mask == 1)

    # test_points = np.argwhere(cropped_image > 0)

    # radial_distances = []
    # for point in test_points:
    #     squared_x_distances = np.power(np.array(sorted_border_points)[:, 0] - point[0], 2)
    #     squared_y_distances = np.power(np.array(sorted_border_points)[:, 1] - point[1], 2)

    #     radial_distances = np.sqrt(squared_x_distances + squared_y_distances)

    #     variance = np.var(radial_distances)

    #     if variance < minimum_variance_radial_distance_center:
    #         minimum_variance_radial_distance_center = variance
    #         minimum_variance_center_coords = (point[0], point[1])

    min_x = np.min(test_points[:, 0])
    max_x = np.max(test_points[:, 0])
    min_y = np.min(test_points[:, 1])
    max_y = np.max(test_points[:, 1])

    squared_x_distances = []
    squared_y_distances = []
    for i in range(min_x, max_x + 1):
        squared_x_distances.append(np.power(np.array(sorted_border_points)[:, 0] - i, 2))
    for j in range(min_y, max_y + 1):
        squared_y_distances.append(np.power(np.array(sorted_border_points)[:, 1] - j, 2))

    for point in test_points:
        radial_distances = np.sqrt(squared_x_distances[point[0] - min_x] + squared_y_distances[point[1] - min_y])

        variance = np.var(radial_distances)

        if variance < minimum_variance_radial_distance_center:
            minimum_variance_radial_distance_center = variance
            minimum_variance_center_coords = (point[0].astype(np.int16), point[1].astype(np.int16))

    return minimum_variance_center_coords



def find_center_v3(cropped_cell_mask):
    dist_map = cv2.distanceTransform(cropped_cell_mask, cv2.DIST_L2, 3)
    return np.argwhere(dist_map == np.max(dist_map))


def find_center_v4(cropped_cell_mask, sorted_border_points=None):
    minimum_variance_radial_distance_center = 100000000000
    minimum_variance_center_coords = (0, 0)

    if sorted_border_points == None:
        sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

    
    skeleton = skeletonize(cropped_cell_mask)
    test_points = np.argwhere(skeleton == 1).astype(np.uint8)
    # vis = deepcopy(cropped_cell_mask)
    # for point in test_points:
    #     vis[point[0], point[1]] = 10
    # plt.imshow(vis)
    # plt.show()

    # test_points = np.argwhere(cropped_image > 0)

    # radial_distances = []
    # for point in test_points:
    #     squared_x_distances = np.power(np.array(sorted_border_points)[:, 0] - point[0], 2)
    #     squared_y_distances = np.power(np.array(sorted_border_points)[:, 1] - point[1], 2)

    #     radial_distances = np.sqrt(squared_x_distances + squared_y_distances)

    #     variance = np.var(radial_distances)

    #     if variance < minimum_variance_radial_distance_center:
    #         minimum_variance_radial_distance_center = variance
    #         minimum_variance_center_coords = (point[0], point[1])

    min_x = np.min(test_points[:, 0])
    max_x = np.max(test_points[:, 0])
    min_y = np.min(test_points[:, 1])
    max_y = np.max(test_points[:, 1])

    squared_x_distances = []
    squared_y_distances = []
    for i in range(min_x, max_x + 1):
        squared_x_distances.append(np.power(np.array(sorted_border_points)[:, 0] - i, 2))
    for j in range(min_y, max_y + 1):
        squared_y_distances.append(np.power(np.array(sorted_border_points)[:, 1] - j, 2))

    for point in test_points:
        radial_distances = np.sqrt(squared_x_distances[point[0] - min_x] + squared_y_distances[point[1] - min_y])

        # variance = np.var(radial_distances)
        variance = np.median(radial_distances)

        if variance < minimum_variance_radial_distance_center:
            minimum_variance_radial_distance_center = variance
            minimum_variance_center_coords = (point[0].astype(np.int16), point[1].astype(np.int16))

    return minimum_variance_center_coords


def find_center_v5(cropped_cell_mask, sorted_border_points=None):
    minimum_median_radial_distance_center = 100000000000
    minimum_median_center_coords = (0, 0)

    if sorted_border_points == None:
        sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

    
    skeleton = skeletonize(cropped_cell_mask)
    test_points = np.argwhere(skeleton == 1).astype(np.uint8)
    median_skeleton_coord = np.median(test_points, axis=0)
    # vis = deepcopy(cropped_cell_mask)
    # for point in test_points:
    #     vis[point[0], point[1]] = 2
    # vis[int(median_skeleton_coord[0]), int(median_skeleton_coord[1])] = 5
    # plt.imshow(vis)
    # plt.show()


    distances = np.linalg.norm(test_points - median_skeleton_coord, axis=1)
    closest_indices = np.argsort(distances)[:3]
    
    test_points = test_points[closest_indices]


    min_x = np.min(test_points[:, 0])
    max_x = np.max(test_points[:, 0])
    min_y = np.min(test_points[:, 1])
    max_y = np.max(test_points[:, 1])

    squared_x_distances = []
    squared_y_distances = []
    for i in range(min_x, max_x + 1):
        squared_x_distances.append(np.power(np.array(sorted_border_points)[:, 0] - i, 2))
    for j in range(min_y, max_y + 1):
        squared_y_distances.append(np.power(np.array(sorted_border_points)[:, 1] - j, 2))

    for point in test_points:
        radial_distances = np.sqrt(squared_x_distances[point[0] - min_x] + squared_y_distances[point[1] - min_y])

        median = np.median(radial_distances)

        if median < minimum_median_radial_distance_center:
            minimum_median_radial_distance_center = median
            minimum_median_center_coords = (point[0].astype(np.int16), point[1].astype(np.int16))

    return minimum_median_center_coords



######################
  #RADIAL DISTANCES
######################

def get_radial_distances(center_coords, sorted_border_points=None, cropped_cell_mask=None):
  """
  Get Radial Distances
  Calculates the radial distances of the border points to the center sorted by clockwise traversal
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation

  Returns:
      NDArray: the radial distances of the border points to the center sorted by clockwise traversal
  """
  if sorted_border_points is None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

  sorted_border_points = np.array(sorted_border_points)

  diffs = sorted_border_points - center_coords
  radial_distances = np.sqrt(np.sum(diffs**2, axis=1))

  return radial_distances



###########################
  #PLOT RADIAL DISTANCES
###########################

def get_radial_distance_plot(cropped_cell_mask, center_coords):
  """
  Get Radial Distance Plot
  Plots the radial distances of the border points sorted by clockwise traversal
  to the center with the least variance in radial distances
  Also plots the mean radial distance
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation

  Returns:
      N/A
  Displays:
      Plot of the radial distances of the border points sorted by clockwise traversal
      to the center with the least variance in radial distances as well as the mean radial distance
  """
  #sorted_border_points = get_border_coords_sorted_by_angle_to_point(cropped_cell_mask, center_coords)
  sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)
  #NOTE - would be better with clockwise border traversal

  radial_distances = np.sqrt(np.power(np.array(sorted_border_points)[:, 0] - center_coords[0], 2) + np.power(np.array(sorted_border_points)[:, 1] - center_coords[1], 2))

  normalized_radial_distances = radial_distances / np.max(radial_distances)

  #mean_radial_distance = np.mean(normalized_radial_distances)
  mean_radial_distance = np.mean(radial_distances)

  # plt.plot(normalized_radial_distances)
  #plt.plot(radial_distances)
  #plt.axhline(y=mean_radial_distance, color='r', linestyle='-')
  #plt.show()




#################################
  #SORT PIXEL BY ANGLE TO POINT
#################################


def get_border_pixels_sorted_by_angle_to_point(cropped_cell_mask, point):
  """
  Get Border Pixels Sorted By Angle to Point
  Returns the border points of the cell segmentation sorted by angle to the point provided
  NOT CURRENTLY IN USE
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    point (tuple): tuple representing the point to sort the border points by

  Returns:
      NDArray: the border points of the cell segmentation sorted by angle to the point provided
  """
  border_points = get_border_coords(cropped_cell_mask)
  angles = {}
  for border_point in border_points:
    angle = np.arctan2(border_point[1] - point[1], border_point[0] - point[0])
    angles[border_point] = angle

  sorted_angles = dict(sorted(angles.items(), key=lambda item: item[1]))
  return list(sorted_angles.keys())









##########################

  #NEW SHAPE PARAMETERS

##########################




################
  #ELONGATION
################

def get_elongation(minimum_area_bounding_box_width=None, minimum_area_bounding_box_height=None, cropped_cell_mask=None):
  """
  Get Elongation
  Source: https://link.springer.com/article/10.1007/s11356-023-26388-5?fromPaywallRec=true
  Elongation = I / L
    I - the shortest axis of the particle’s minimum bounding box
    L - the longest axis of the particle’s minimum bounding box
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      float: the elongation metric of the segmentation
  """
  if minimum_area_bounding_box_width is None or minimum_area_bounding_box_height is None:
    minimum_area_bounding_box_vertices = get_minimum_bounding_box(cropped_cell_mask)

    top_left, top_right, bottom_left, bottom_right = minimum_area_bounding_box_vertices
    top_length = np.sqrt(np.power(top_left[0] - top_right[0], 2) + np.power(top_left[1] - top_right[1], 2))
    side_length = np.sqrt(np.power(bottom_left[0] - top_left[0], 2) + np.power(bottom_left[1] - top_left[1], 2))
    long_side = max(top_length, side_length)
    short_side = min(top_length, side_length)

    elongation = short_side / long_side

    return elongation

  else:
    long_side = max(minimum_area_bounding_box_width, minimum_area_bounding_box_height)
    short_side = min(minimum_area_bounding_box_width, minimum_area_bounding_box_height)

    elongation = short_side / long_side

    return elongation



###############################
  #ELLIPTISITY and ECCENTRICITY
###############################

def get_elliptisity(major_axis_length=None, minor_axis_length=None, cropped_cell_mask=None):
  """
  Get Elliptisity
  Elliptisity = Major Axis Length / Minor Axis Length
  Author(s): ####

  Args:
    mask (np.ndarray): array of variable shape representing a binary mask of the cell

  Returns:
      float: the elliptisity metric of the segmentation
  """

  if major_axis_length is None or minor_axis_length is None:
    major_axis_length, minor_axis_length, _ = get_major_minor_axis_and_theta(cropped_cell_mask)

  elliptisity = major_axis_length / minor_axis_length

  return elliptisity


def get_eccentricity(major_axis_length=None, minor_axis_length=None, cropped_cell_mask=None):
  """
  Get Eccentricity
  Eccentricity = sqrt(1 - (minor_axis_length / major_axis_length)^2)
  Author(s): ####

  Args:
    mask (np.ndarray): array of variable shape representing a binary mask of the cell

  Returns:
      float: the eccentricity metric of the segmentation
  """

  if major_axis_length is None or minor_axis_length is None:
    major_axis_length, minor_axis_length, _ = get_major_minor_axis_and_theta(cropped_cell_mask)

  eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length)**2)

  return eccentricity



################
  #COMPACTNESS
################

def get_compactness(segmentation_perimeter=None, segmentation_area=None, cropped_cell_mask=None):
  """
  Get Compactness
  Source: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/3661/0000/Effects-of-image-resolution-and-segmentation-method-on-automated-mammographic/10.1117/12.348654.short#_=_
  Compactness = P^2 / A
    P - perimeter of segmentation
    A - area of segmentation
  Note: Same as Circularity Three
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a binary mask of the cell

  Returns:
      float: the compactness metric of the segmentation
  """

  if segmentation_perimeter == None:
    segmentation_perimeter = get_perimeter_v3(cropped_cell_mask)
  if segmentation_area == None:
    segmentation_area = area(cropped_cell_mask)

  compactness = (segmentation_perimeter**2) / segmentation_area

  return compactness

def get_compactness_two(cropped_cell_mask, center_coords, cell_area):
  """
  Get Compactness
  Source: https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-3.0.0/modules/measurement.html
  Compactness = mean squared distance of objects pixels from center divided by area
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation
    cell_area (int): area of the cell segmentation

  Returns:
      float: the compactness metric of the segmentation
  """

  points = np.argwhere(cropped_cell_mask > 0)
    
  # Squared distances to center
  diff = points - center_coords  # shape (N, 2)
  squared_distances = np.sum(diff**2, axis=1)  # sum squared over x,y
  
  mean_squared_distances_to_center = np.sum(squared_distances)

  compactness = (mean_squared_distances_to_center / cell_area**2)

  return compactness





################
  #CIRCULARITY
################

def get_circularity(segmentation_perimeter=None, segmentation_area=None, cropped_cell_mask=None):
  """
  Get Circularity
  Source: https://link.springer.com/article/10.1007/s11356-023-26388-5?fromPaywallRec=true
  Circularity = 4πA/P^2
    A - area of segmentation
    P - perimeter of segmentation
  This is the ratio of the circumference of a circle of the same area as the
  segmentation to the actual circumference of the segmentation
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a binary mask of the cell

  Returns:
      float: the circularity metric of the segmentation
  """

  if segmentation_area == None:
    segmentation_area = area(cropped_cell_mask)
  if segmentation_perimeter == None:
    segmentation_perimeter = get_perimeter_v3(cropped_cell_mask)

  circularity = (4 * np.pi * segmentation_area) / (segmentation_perimeter**2)

  return circularity

def get_circularity_two(segmentation_area=None, convex_hull_perimeter=None, cropped_cell_mask=None):
  """
  Get Circularity Two
  Source: http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
  Circularity = 4πA/Pc^2
    A - area of segmentation
    Pc - perimeter of convex hull
  This is the ratio of the area of the segmentation to the area of a circle with the same
  convex perimeter
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a binary mask of the cell

  Returns:
      float: the second circularity metric of the segmentation
  """

  if segmentation_area == None:
    segmentation_area = area(cropped_cell_mask)
  if convex_hull_perimeter == None:
    convex_hull_perimeter = get_convex_hull_perimeter(cropped_cell_mask)

  circularity = (4 * np.pi * segmentation_area) / (convex_hull_perimeter**2)

  return circularity

def get_circularity_three(segmentation_perimeter=None, segmentation_area=None, cropped_cell_mask=None):
  """
  Get Circularity Three
  Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=251116
  Circularity = P^2/A
    P - perimeter of segmentation
    A - area of segmentation
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a binary mask of the cell

  Returns:
      float: the third circularity metric of the segmentation
  """

  if segmentation_perimeter == None:
    segmentation_perimeter = get_perimeter_v3(cropped_cell_mask)
  if segmentation_area == None:
    segmentation_area = area(cropped_cell_mask)

  circularity = (segmentation_perimeter**2) / segmentation_area

  return circularity

def get_circularity_four(center_coords, radial_distances=None, cropped_cell_mask=None):
    """
    Get Circularity Four
    Source: https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1118/1.597707?saml_referrer
    Circularity = mean radial distance of boundary / standard deviation of radial distance of boundary
    Author(s): ####

    Args:
        cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
        center_coords (tuple): coordinates of the center of the cell segmentation

    Returns:
        float: the fourth circularity metric of the segmentation
    """
    if radial_distances is None:
        sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

        radial_distances = np.sqrt(np.power(np.array(sorted_border_points)[:, 0] - center_coords[0], 2) + np.power(np.array(sorted_border_points)[:, 1] - center_coords[1], 2))

    normalized_radial_distances = radial_distances / np.max(radial_distances)

    mean_radial_distance = np.mean(normalized_radial_distances)
    std_radial_distance = np.std(normalized_radial_distances)


    if std_radial_distance == 0:
        raise ValueError('STD Radial Distance is zero, can\'t calculate circularity (4)')        
    circularity = mean_radial_distance / std_radial_distance

    return circularity




##############
  #CONVEXITY
##############

def get_convexity(segmentation_area=None, convex_hull_area=None, cropped_cell_mask=None):
  """
  Get Convexity
  Source: https://link.springer.com/article/10.1007/s11356-023-26388-5?fromPaywallRec=true
  Convexity = A / Ac
    A - area of the segmentation
    Ac - area of the convex hull of the segmentation
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a binary mask of the cell

  Returns:
      float: the convexity metric of the segmentation
  """
  if segmentation_area == None:
    segmentation_area = area(cropped_cell_mask)
  if convex_hull_area == None:
    convex_hull_area = get_convex_hull_area(cropped_cell_mask)

  convexity = segmentation_area / convex_hull_area

  return convexity


def get_convexity_two(segmentation_perimeter=None, convex_hull_perimeter=None, cropped_cell_mask=None):
  """
  Get Convexity Two
  Source: http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
  Convexity = Pc / P
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

  Returns:
      float: the second convexity metric of the segmentation
  """

  if segmentation_perimeter == None:
    segmentation_perimeter = get_perimeter_v3(cropped_cell_mask)
  if convex_hull_perimeter == None:
    convex_hull_perimeter = get_convex_hull_perimeter(cropped_cell_mask)

  convexity = convex_hull_perimeter / segmentation_perimeter

  return convexity





##############
  #ROUGHNESS
##############

def get_roughness(segmentation_perimeter=None, convex_hull_perimeter=None, cropped_cell_mask=None):
  """
  Get Roughness
  Source: https://link.springer.com/article/10.1007/s11356-023-26388-5?fromPaywallRec=true
  Roughness = P / Pc
    P - perimeter of the segmentation
    Pc - perimeter of the convex hull of the segmentation
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a binary mask of the cell

  Returns:
      float: the roughness metric of the segmentation
  """
  if segmentation_perimeter == None:
    segmentation_perimeter = get_perimeter_v3(cropped_cell_mask)
  if convex_hull_perimeter == None:
    convex_hull_perimeter = get_convex_hull_perimeter(cropped_cell_mask)

  roughness = segmentation_perimeter / convex_hull_perimeter

  return roughness

def get_roughness_two(center_coords, sorted_border_points=None, radial_distances=None, cropped_cell_mask=None):
  """
  Get Roughness Two
  Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=251116
  Roughness = look in notes/source
    Takes the average over intervals that sum of the absolute difference between the radial
    distance of 3 adjacent border points
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation

  Returns:
      float: the second roughness metric of the segmentation
  """
  if sorted_border_points is None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)
  if radial_distances is None:
    radial_distances = np.sqrt(np.power(np.array(sorted_border_points)[:, 0] - center_coords[0], 2) + np.power(np.array(sorted_border_points)[:, 1] - center_coords[1], 2))

  normalized_radial_distances = radial_distances / np.max(radial_distances)

  intervals = int(len(sorted_border_points) / 3)
  intervals_size = int(len(normalized_radial_distances) / intervals)

  total_roughness = 0
  for i in range(intervals):
    start = i * intervals_size
    end = (i + 1) * intervals_size
    interval_roughness = 0
    for j in range(start, end):
      if j + 2 < len(normalized_radial_distances):
        interval_roughness += abs(normalized_radial_distances[j] - normalized_radial_distances[j + 1])

    total_roughness += interval_roughness

  avg_roughness = total_roughness / intervals

  return avg_roughness


def get_roughness_three(center_coords, border_points=None, cropped_cell_mask=None):
  """
  Get Roughness Three
  Source: https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1118/1.597707?saml_referrer
  Roughness - number of angles w/ multiple points / total number of angles
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation

  Returns:
      float: the third roughness metric of the segmentation
  """

  if border_points is None:
    border_points = get_border_coords(cropped_cell_mask)

  angles = {}
  for point in border_points:
    angle = np.arctan2(point[1] - center_coords[1], point[0] - center_coords[0])
    if angle in angles.keys():
      angles[angle] += 1
    else:
      angles[angle] = 1

  angles_with_multiple_points = 0
  for angle in angles.keys():
    if angles[angle] > 1:
      angles_with_multiple_points += 1

  roughness = angles_with_multiple_points / len(angles)

  return roughness


def count_mean_crossings(center_coords, radial_distances=None, cropped_cell_mask=None):
  """
  Count Mean Crossings
  Counts the number of times that the radial distance of the border points sorted by clockwise traversal
  to the center moves from above the mean radial distance to below the mean radial distance or vice versa
  Note: could be used for a roughness measure
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation

  Returns:
      int: the number of times the radial distance cross the mean
  """
  if radial_distances is None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

    radial_distances = get_radial_distances(center_coords, sorted_border_points)

  mean_radial_distance = np.mean(radial_distances)

  crossings = 0
  for i in range(1, len(radial_distances)):
    if radial_distances[i - 1] > mean_radial_distance and radial_distances[i] < mean_radial_distance:
      crossings += 1
    elif radial_distances[i - 1] < mean_radial_distance and radial_distances[i] > mean_radial_distance:
      crossings += 1

  return crossings


def count_mean_crossings_v2(center_coords, radial_distances=None, cropped_cell_mask=None):
  """
  Count Mean Crossings
  Counts the number of times that the radial distance of the border points sorted by clockwise traversal
  to the center moves from above the mean radial distance to below the mean radial distance or vice versa
  Note: could be used for a roughness measure
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation

  Returns:
      int: the number of times the radial distance cross the mean
  """
  if radial_distances is None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

    radial_distances = np.sqrt(np.power(np.array(sorted_border_points)[:, 0] - center_coords[0], 2) + np.power(np.array(sorted_border_points)[:, 1] - center_coords[1], 2))

  mean_radial_distance = np.mean(radial_distances)

  diff = radial_distances - mean_radial_distance
  # Signs of differences (+/-)
  signs = np.sign(diff)
  # Replace zeros with previous sign to avoid false crossings at exact mean
  signs[signs == 0] = 1  

  # Count sign changes between consecutive points
  crossings = np.sum(signs[1:] != signs[:-1])

  return crossings





#############################
  #RADIAL DISTANCE MEASURES
#############################

def get_mean_radial_distance(center_coords, radial_distances=None, cropped_cell_mask=None):
  """
  Get Mean Radial Distance
  Returns the mean of the radial distances to the center
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation

  Returns:
      float: the mean of the radial distances to the center
  """
  if radial_distances is None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

    radial_distances = get_radial_distances(center_coords, sorted_border_points)


  normalized_radial_distances = radial_distances / np.max(radial_distances)

  mean_radial_distance = np.mean(normalized_radial_distances)

  return mean_radial_distance


def get_std_radial_distance(center_coords, radial_distances=None, cropped_cell_mask=None):
  """
  Get STD Radial Distance
  Returns the std of the radial distances to the center
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation

  Returns:
      float: the std of the radial distances to the center
  """
  if radial_distances is None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

    radial_distances = np.sqrt(np.power(np.array(sorted_border_points)[:, 0] - center_coords[0], 2) + np.power(np.array(sorted_border_points)[:, 1] - center_coords[1], 2))

  normalized_radial_distances = radial_distances / np.max(radial_distances)

  std_radial_distance = np.std(normalized_radial_distances)

  return std_radial_distance


def get_entropy_of_radial_distance(center_coords, radial_distances=None, cropped_cell_mask=None):
  """
  Get Entropy of Radial Distance
  Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=251116
  Entropy = -summation from k = 1 to 100, probability of k * log(probability of k)
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation

  Returns:
      float: the entorpy of the radial distances to the center
  """

  if radial_distances is None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)

    radial_distances = get_radial_distances(center_coords, sorted_border_points)

  normalized_radial_distances = radial_distances / np.max(radial_distances)

  probs = []
  intervals = 100
  for i in range(intervals):
    in_range = np.where((normalized_radial_distances >= i / intervals) & (normalized_radial_distances < (i + 1) / intervals))[0]
    probs.append(len(in_range) / len(normalized_radial_distances))


  # plt.hist(radial_distances, bins='auto', range=(np.min(radial_distances) - 10, np.max(radial_distances) + 10))
  # plt.show()

  entropy = 0
  for prob in probs:
    if prob != 0:
      entropy -= prob * np.log2(prob)

  return entropy

def get_entropy_of_radial_distance_v2(center_coords, radial_distances=None, cropped_cell_mask=None):
  """
  Get Entropy of Radial Distance
  Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=251116
  Entropy = -summation from k = 1 to 100, probability of k * log(probability of k)
  Author(s): ####

  Args:
    cropped_cell_mask (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.
    center_coords (tuple): coordinates of the center of the cell segmentation

  Returns:
      float: the entorpy of the radial distances to the center
  """

  if radial_distances is None:
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask)
    radial_distances = get_radial_distances(center_coords, sorted_border_points)

  normalized_radial_distances = radial_distances / np.max(radial_distances)

  intervals = 100
  counts, _ = np.histogram(normalized_radial_distances, bins=intervals, range=(0, 1))

  probs = counts / counts.sum()

  # Filter zero probabilities to avoid log2(0)
  probs_nonzero = probs[probs > 0]

  entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))

  return entropy





###############################
  #HARALICK TEXTURE FEATURES
###############################

def get_haralick_features(cropped_image):
    """
    Get Haralick Features
    Source: https://mahotas.readthedocs.io/en/latest/, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4309314
    Returns 4x14 vector of haralick features in 4 directions and a 1x14 vector of mean haralick features
    Features:
        1) Angular Second Moment (ASM)
        2) Contrast
        3) Correlation
        4) Sum of Squares: Variance
        5) Inverse Difference Moment (IDM)
        6) Sum Average
        7) Sum Variance
        8) Sum Entropy
        9) Entropy
        10) Difference Variance
        11) Difference Entropy
        12) Information Measure of Correlation 1
        13) Information Measure of Correlation 2
        14) Maximal Correlation Coefficient
    Author(s): ####

    Args:
        cropped_image (np.ndarray): array of variable shape representing a grayscale image cropped to the cell mask.

    Returns:
        np.ndarray: 4x14 vector of haralick features in 4 directions
        np.ndarray: 1x14 vector of mean haralick features
    """

    features = mahotas.features.haralick(cropped_image, compute_14th_feature=True, ignore_zeros=False)
    mean_features = np.mean(features, axis=0)
    return features, mean_features




##############
  #SPIKINESS
##############

def get_spikiness_by_perimeter_and_area(cell_perimeter=None, cell_area=None, cropped_cell_mask=None):
  """
  Get Spikiness by Perimeter and Area
  Calculates the spikiness of a cell mask by taking the ratio of its perimeter to its area.
  The larger this value (the smaller the area of the cell compared to its perimeter), the more spiky the cell is.
  Motivation:
    -We generally think of shapes with fewer sides as more spiky (i.e. a triangle is spikier than a circle (consider circle to have infinite sides) or an octagon)
    -We also know that, given the same perimeter, shapes with fewer sides (that are therefore less spiky) have a larger area.
      -Example: Perimeter = 10
        -Area of Circle(inf sides): 7.96
        -Area of Decagon(10 sides): 7.69
        -Area of Octagon(8 sides): 7.54
        -Area of Hexagon(6 sides): 7.22
        -Area of Pentagon(5 sides): 6.88
        -Area of Square(4 sides): 6.25
        -Area of Triangle(3 sides): 4.81 (equilateral)
    -Therefore, by taking the ratio of perimeter to area, we know that masks that have a larger ratio
      will have smaller areas compared to perimeter and therefore will have fewer sides and be more "spiky"
  Author(s): ####

  Args:
    mask (np.ndarray): Array of shape (W,H) representing a binary mask of the cell. 1 = cell, 0 = background.

  Returns:
      float: The spikiness of the cell mask representated as a ratio of its perimeter to its area.
  """

  if cell_perimeter == None:
    cell_perimeter = get_perimeter_v3(cropped_cell_mask)
  if cell_area == None:
    cell_area = area_v2(cropped_cell_mask)

  return cell_perimeter / cell_area




def get_image_parameters(img, ann):
    parameters = {}
    for i in np.unique(ann):
        if i != 0:
            cell_mask = get_cell_mask(ann, i)
            cropped_cell_mask, image_crop, crop_coords = crop_to_mask(cell_mask, img)
            cropped_image = get_mask_pixel_intensities(image_crop, cropped_cell_mask) #25s up to here
    

            #Border Points (and Clockwise Traversal)
            border_points = get_border_coords(cropped_cell_mask)
            sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask, border_points, plot=False)

            #Cell Center
            cell_center = find_center_v5(cropped_cell_mask)

            #Perimeter and Area
            # cell_perimeter = get_perimeter_new_v2(cropped_cell_mask, sorted_border_points)
            cell_perimeter = get_perimeter_v3(sorted_border_points)
            cell_area = area_v2(cropped_cell_mask)

            #Radial Distances
            radial_distances = get_radial_distances(cell_center, sorted_border_points)

            ##Major, Minor Axes (and Theta)
            major_axis, minor_axis, theta = get_major_minor_axis_and_theta(cropped_cell_mask)

            ##Convex Hull
            # convex_hull_vertices, convex_hull_edges = get_convex_hull(cropped_cell_mask, border_points)
            # convex_hull_area = get_convex_hull_area(cropped_cell_mask, convex_hull_edges)
            # convex_hull_perimeter = get_convex_hull_perimeter(cropped_cell_mask, convex_hull_edges)
            convex_hull_vertices, convex_hull_area, convex_hull_perimeter = get_convex_hull_v2(border_points)

            ##Minimum Bounding Box
            # minimum_area_bounding_box_vertices = get_minimum_bounding_box(cropped_cell_mask)
            minimum_area_bounding_box_vertices, minimum_area_bounding_box_width, minimum_area_bounding_box_height = get_minimum_bounding_box_v2(convex_hull_vertices, cropped_cell_mask)

            #Radii of Largest Inscribed and Smallest Encompassing Circles
            largest_inscribed_circle_radius = radius_of_largest_inscribed_circle(cropped_cell_mask)
            smallest_enclosing_circle_radius = radius_of_smallest_enclosing_circle(cropped_cell_mask)
            
            #Number of Surrounding Cells
            surrounding_cells, num_surrounding_cells = get_surrounding_cells(cell_mask, ann, crop_coords, cell_index=i) #29s up to here
            
            spikiness_metric = get_spikiness_by_perimeter_and_area(cell_perimeter, cell_area)
            elongation = get_elongation(minimum_area_bounding_box_width, minimum_area_bounding_box_height)
            eccentricity = get_eccentricity(major_axis, minor_axis, cropped_cell_mask)
            elliptisity = get_elliptisity(major_axis, minor_axis, cropped_cell_mask)
            compactness = get_compactness(cell_perimeter, cell_area)
            compactness_two = get_compactness_two(cropped_cell_mask, cell_center, cell_area)
            circularity_one = get_circularity(cell_perimeter, cell_area)
            circularity_two = get_circularity_two(cell_area, convex_hull_perimeter)
            circularity_three = get_circularity_three(cell_perimeter, cell_area)
            circularity_four = get_circularity_four(cell_center, radial_distances)
            convexity_one = get_convexity(cell_area, convex_hull_area)
            convexity_two = get_convexity_two(cell_perimeter, convex_hull_perimeter)
            roughness_one = get_roughness(cell_perimeter, convex_hull_perimeter)
            roughness_two = get_roughness_two(cell_center, sorted_border_points, radial_distances)
            #REMOVED: roughness_three
            mean_crossings = count_mean_crossings_v2(cell_center, radial_distances)
            mean_radial_distance = get_mean_radial_distance(cell_center, radial_distances)
            std_radial_distance = get_std_radial_distance(cell_center, radial_distances)
            entropy_of_radial_distance = get_entropy_of_radial_distance_v2(cell_center, radial_distances)
            haralick_features, mean_haralick_features = get_haralick_features(cropped_image) # 48s

            # #PIXEL METRICS
            
            mean_intensity, std_intensity, max_intensity, min_intensity, intensity_range = calculate_pixel_intensity_metrics(cropped_image, False)
            
            # layer_avgs, gradient_metric = calculate_gradient_layer_by_layer(cropped_image)
            layer_avgs, gradient_metric = calculate_gradient_layer_by_layer_v2(cropped_image, cropped_cell_mask, border_points) # 3s

            parameters[i] = {
                'Perimeter': cell_perimeter,
                'Area': cell_area,
                'Convex Hull Area': convex_hull_area,
                'Convex Hull Perimeter': convex_hull_perimeter,
                'Bounding Box Height': minimum_area_bounding_box_height,
                'Bounding Box Width': minimum_area_bounding_box_width,
                'Major Axis': major_axis,
                'Minor Axis': minor_axis,
                'Theta': theta,
                'Min Enclosing Circle Radius': smallest_enclosing_circle_radius,
                'Max Inscribed Circle Radius': largest_inscribed_circle_radius,
                'Num Surrounding Cells': num_surrounding_cells,
                'Spikiness': spikiness_metric,
                'Elongation': elongation,
                'Eccentricity': eccentricity,
                'Elliptisity': elliptisity,
                'Compactness (1)': compactness,
                'Compactness (2)': compactness_two,
                'Circularity (1)': circularity_one,
                'Circularity (2)': circularity_two,
                'Circularity (3)': circularity_three,
                'Circularity (4)': circularity_four,
                'Convexity (1)': convexity_one,
                'Convexity (2)': convexity_two,
                'Roughness (1)': roughness_one,
                'Roughness (2)': roughness_two,
                'Mean Radial Distance Crossings': mean_crossings,
                'Mean Radial Distance': mean_radial_distance,
                'STD Radial Distance': std_radial_distance,
                'Entropy of Radial Distance': entropy_of_radial_distance,
                'Mean Intensity': mean_intensity,
                'STD_Intensity': std_intensity,
                'Max Intensity': max_intensity,
                'Min Intensity': min_intensity,
                'Intensity Range': intensity_range,
                'Intensity Gradient Metric': gradient_metric
            }

            dir_index = 0
            for direction_features in haralick_features:
                dir_index += 1
                parameters[i][f'Haralick - Angular Second Moment (Direction {dir_index})'] = direction_features[0]
                parameters[i][f'Haralick - Contrast (Direction {dir_index})'] = direction_features[1]
                parameters[i][f'Haralick - Correlation (Direction {dir_index})'] = direction_features[2]
                parameters[i][f'Haralick - Sum of Squares: Variance (Direction {dir_index})'] = direction_features[3]
                parameters[i][f'Haralick - Inverse Difference Moment (Direction {dir_index})'] = direction_features[4]
                parameters[i][f'Haralick - Sum Average (Direction {dir_index})'] = direction_features[5]
                parameters[i][f'Haralick - Sum Variance (Direction {dir_index})'] = direction_features[6]
                parameters[i][f'Haralick - Sum Entropy (Direction {dir_index})'] = direction_features[7]
                parameters[i][f'Haralick - Entropy (Direction {dir_index})'] = direction_features[8]
                parameters[i][f'Haralick - Difference Variance (Direction {dir_index})'] = direction_features[9]
                parameters[i][f'Haralick - Difference Entropy (Direction {dir_index})'] = direction_features[10]
                parameters[i][f'Haralick - Information Measure Correlation 1 (Direction {dir_index})'] = direction_features[11]
                parameters[i][f'Haralick - Information Measure Correlation 2 (Direction {dir_index})'] = direction_features[12]
                parameters[i][f'Haralick - Maximal Correlation Coefficient (Direction {dir_index})'] = direction_features[13]

            parameters[i][f'Haralick - Angular Second Moment (Mean)'] = mean_haralick_features[0]
            parameters[i][f'Haralick - Contrast (Mean)'] = mean_haralick_features[1]
            parameters[i][f'Haralick - Correlation (Mean))'] = mean_haralick_features[2]
            parameters[i][f'Haralick - Sum of Squares: Variance (Mean)'] = mean_haralick_features[3]
            parameters[i][f'Haralick - Inverse Difference Moment (Mean)'] = mean_haralick_features[4]
            parameters[i][f'Haralick - Sum Average (Mean)'] = mean_haralick_features[5]
            parameters[i][f'Haralick - Sum Variance (Mean)'] = mean_haralick_features[6]
            parameters[i][f'Haralick - Sum Entropy (Mean)'] = mean_haralick_features[7]
            parameters[i][f'Haralick - Entropy (Mean)'] = mean_haralick_features[8]
            parameters[i][f'Haralick - Difference Variance (Mean)'] = mean_haralick_features[9]
            parameters[i][f'Haralick - Difference Entropy (Mean)'] = mean_haralick_features[10]
            parameters[i][f'Haralick - Information Measure Correlation 1 (Mean)'] = mean_haralick_features[11]
            parameters[i][f'Haralick - Information Measure Correlation 2 (Mean)'] = mean_haralick_features[12]
            parameters[i][f'Haralick - Maximal Correlation Coefficient (Mean)'] = mean_haralick_features[13]

    return parameters




def get_segmentation_parameters_for_shape_score(segmentations, seg_index, img):
    
    cell_mask = get_cell_mask(segmentations, seg_index)
    cropped_cell_mask, image_crop, crop_coords = crop_to_mask(cell_mask, img)


    #Border Points (and Clockwise Traversal)
    border_points = get_border_coords(cropped_cell_mask)
    sorted_border_points = get_clockwise_border_traversal(cropped_cell_mask, border_points, plot=False)

    #Cell Center
    cell_center = find_center_v5(cropped_cell_mask)

    #Perimeter and Area
    cell_perimeter = get_perimeter_v3(sorted_border_points)
    cell_area = area_v2(cropped_cell_mask)

    ##Convex Hull
    convex_hull_vertices, convex_hull_area, convex_hull_perimeter = get_convex_hull_v2(border_points)

    ##Minimum Bounding Box
    minimum_area_bounding_box_vertices, minimum_area_bounding_box_width, minimum_area_bounding_box_height = get_minimum_bounding_box_v2(convex_hull_vertices, cropped_cell_mask)
    
    spikiness_metric = get_spikiness_by_perimeter_and_area(cell_perimeter, cell_area)
    elongation = get_elongation(minimum_area_bounding_box_width, minimum_area_bounding_box_height)
    compactness_two = get_compactness_two(cropped_cell_mask, cell_center, cell_area)
    circularity_two = get_circularity_two(cell_area, convex_hull_perimeter)
    convexity_one = get_convexity(cell_area, convex_hull_area)
    roughness_one = get_roughness(cell_perimeter, convex_hull_perimeter)


    parameters = {
        'Perimeter': cell_perimeter,
        'Area': cell_area,
        'Spikiness': spikiness_metric,
        'Elongation': elongation,
        'Compactness': compactness_two,
        'Circularity': circularity_two,
        'Convexity': convexity_one,
        'Roughness': roughness_one,
    }

    return parameters



