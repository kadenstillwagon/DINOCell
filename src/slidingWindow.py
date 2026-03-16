import cv2
import numpy as np

class SlidingWindowHelper:
    def __init__(self, crop_size: int, overlap_size: int):
        self.crop_size = crop_size
        self.overlap_size = overlap_size
    
    def seperate_into_crops_v2(self, img):
        # Initialize a list to store cropped images
        cropped_images = []
        orig_regions = []

        height, width = img.shape

        for y in range(0, height, self.crop_size - self.overlap_size * 2):
            for x in range(0, width, self.crop_size - self.overlap_size * 2):
                # Calculate crop boundaries
                x_start = x
                x_end = x + self.crop_size
                y_start = y
                y_end = y + self.crop_size

                if x_end > width:
                    x_start = width - self.crop_size
                    x_end = width
                if y_end > height:
                    y_start = height - self.crop_size
                    y_end = height

                # Extract the crop
                crop = img[y_start:y_end, x_start:x_end]

                #get the crop_coords
                orig_region = (x_start, y_start, x_end, y_end)

                # Append the cropped image to the list
                cropped_images.append(crop)
                orig_regions.append(orig_region)
        
        return cropped_images, orig_regions

    def combine_crops_v2(self, orig_size, cropped_images, orig_regions):
        predictions_sum = np.zeros(orig_size, dtype=np.float32)
        predictions_num = np.zeros(orig_size, dtype=np.float32)
        for crop, region in zip(cropped_images, orig_regions):
            x, y, x_end, y_end = region
            predictions_sum[y:y_end, x:x_end] += crop
            predictions_num[y:y_end, x:x_end] += 1
        
        output_img = predictions_sum / predictions_num
        # plt.imshow(output_img)
        # plt.savefig('pipeline_test.png')
        # plt.close()
        return output_img
