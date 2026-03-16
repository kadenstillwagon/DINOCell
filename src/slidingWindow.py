import cv2
import numpy as np
import matplotlib.pyplot as plt

class SlidingWindowHelper:
    def __init__(self, crop_size: int, overlap_size: int):
        self.crop_size = crop_size
        self.overlap_size = overlap_size

    def seperate_into_crops(self, img):
        #mirror the image such that the edges are repeated around the overlap region
        img_mirrored = cv2.copyMakeBorder(img, self.overlap_size, self.overlap_size, self.overlap_size, self.overlap_size, cv2.BORDER_REFLECT)

        # Get the image dimensions
        height, width = img_mirrored.shape

        # Initialize a list to store cropped images
        cropped_images = []
        orig_regions = []
        crop_unique_region = (self.overlap_size, self.overlap_size, self.crop_size - 2 * self.overlap_size, self.crop_size - 2 * self.overlap_size)

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

                # Extract the crop with mirrored edges
                crop = img_mirrored[y_start:y_end, x_start:x_end]

                #get the unique portion of the crop
                orig_region = (x_start + self.overlap_size, y_start + self.overlap_size, self.crop_size - 2 * self.overlap_size, self.crop_size - 2 * self.overlap_size)

                # Append the cropped image to the list
                cropped_images.append(crop)
                orig_regions.append(orig_region)
        
        return cropped_images, orig_regions, crop_unique_region

    
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

    def combine_crops(self, orig_size, cropped_images, orig_regions, crop_unique_region):
        output_img = np.zeros(orig_size, dtype=np.float32)
        for crop, region in zip(cropped_images, orig_regions):
            x, y, w, h = region
            x -= self.overlap_size
            y -= self.overlap_size
            # if crop.shape != (256, 256):
            #     continue
            unique_region = crop[crop_unique_region[0]:crop_unique_region[0] + crop_unique_region[2], crop_unique_region[1]:crop_unique_region[1] + crop_unique_region[3]]
            output_img[y:y+h, x:x+w] = unique_region
            # output_img[y:y+h, x:x+w] = random.random() * 255
        return output_img

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
