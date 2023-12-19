'''
Clean raw image pulls (form manga demon)

autosplit according to presence of whitespace
Delete only whitespace files

# Example usage
md = md_image_cleaner(in_directory = 'raws/issues/raw_images',
    out_directory = 'split_images')
md.split_raws()
md.clean_splits()

'''


import os
import numpy as np
import pytesseract
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
import json

from .splits import natural_sort_key


def is_close_to_white(row, tolerance):
    return np.all(np.all(row >= 255 - tolerance, axis=-1))

def is_image_blank(image_path, tolerance):
    with Image.open(image_path) as img:
        data = np.array(img)
    return np.all(is_close_to_white(data, tolerance))

class md_image_cleaner:

    def __init__(self, 
                 in_directory = 'raw_chapters/english199/raw_images/',
                 out_directory = 'scratch'):

        self.in_directory = in_directory
        self.out_directory = out_directory
        self.split_index = 0

    def split_raws(self):
        files = os.listdir(self.in_directory)
        sorted_filenames = sorted(files, key=natural_sort_key)
        for filename in sorted_filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(self.in_directory, filename)
                self._split_image(image_path)

    def _split_image(self,image_path):
        """
        Split an image at specific points.

        :param image_path: Path to the original image.
        :param split_points: List of points (heights in pixels) where the image should be split.
        :param output_folder: Folder to save the split images.
        """
        
        with Image.open(image_path) as img:
            split_points = self._get_split_index(img)
            # Get the dimensions of the image
            img_width, img_height = img.size

            # Initialize the start height for the first segment
            start_height = 0

            # Iterate over the split points
            for split_point in split_points:
                # Ensure the split point is within the image height
                if split_point > img_height:
                    split_point = img_height

                # Crop the image
                cropped_img = img.crop((0, start_height, img_width, split_point))

                # Save the cropped image
                cropped_img.save(f"{self.out_directory}/page_num_{self.split_index}.jpeg")

                # Update variables for the next segment
                start_height = split_point
                self.split_index += 1

            # Process the last segment if there's any remainder
            if start_height < img_height:
                cropped_img = img.crop((0, start_height, img_width, img_height))
                cropped_img.save(f"{self.out_directory}/page_num_{self.split_index}.jpeg")
                self.split_index+=1

    @staticmethod
    def _get_split_index(img, max_splits=20, thold = 500):
        """
        Split an image at the middle of the largest continuous spaces of the most frequent color in a vertical line.

        :param image_path: Path to the original image.
        :param max_splits: Maximum number of splits to make, based on the largest continuous spaces.
        :return: List of split positions (midpoints of the largest continuous spaces).
        """

        # Convert the image to a numpy array
        data = np.array(img)

        #data = base_data[first_split:,:,:]

        # Select a vertical line (1 pixel wide) from the image
        line = data[:, data.shape[1] // 2, :]

        # Determine the most frequent color on the line
        colors, counts = np.unique(line.reshape(-1, line.shape[-1]), axis=0, return_counts=True)
        most_frequent_color = colors[counts.argmax()]

        # Find positions where the color matches the most frequent color
        matches = np.all(line == most_frequent_color, axis=1)

        # Find start and end positions of matching sequences
        diff = np.diff(matches.astype(int))
        start_positions = np.where(diff == 1)[0] + 1
        end_positions = np.where(diff == -1)[0]

        # Handle the case where the sequence starts from the beginning
        if matches[0]:
            start_positions = np.insert(start_positions, 0, 0)

        # Handle the case where the sequence ends at the end
        if matches[-1]:
            end_positions = np.append(end_positions, len(matches) - 1)

        # Calculate the lengths of the continuous spaces
        lengths = end_positions - start_positions
        # Find the midpoints of the largest continuous spaces
        largest_spaces_indices = np.argsort(lengths)[-max_splits:]
        midpoints = (start_positions[largest_spaces_indices] + end_positions[largest_spaces_indices]) // 2
        lengths =  end_positions[largest_spaces_indices] - start_positions[largest_spaces_indices]
        midpoints = [ x for x,l in zip(midpoints,lengths) if float(l) >= float(thold)]
        return sorted(midpoints) 


    def clean_splits(self):
        """
        Roll through directory and clean all auto-split images
        """
        files = os.listdir(self.out_directory)
        sorted_filenames = sorted(files, key=natural_sort_key)
        for filename in sorted_filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(self.out_directory, filename)
                self.trim_whitespace_from_images(image_path)
                self.delete_blank_images(image_path)


    def trim_whitespace_from_images(self, image_path, tolerance=50, border=10):
        """
        Trim whitespace from the top and bottom of all images in a directory, leaving a specified border.

        :param directory: Path to the directory containing images.
        :param tolerance: Tolerance value to consider a pixel as 'white' or close to white.
        :param border: Border size to leave around the trimmed image.
        """
        filename = image_path.split(os.path.sep)[-1]
        with Image.open(image_path) as img:
            data = np.array(img)

            non_white_rows = [i for i, row in enumerate(data) if not is_close_to_white(row, tolerance)]
            top_trim = max(min(non_white_rows) - border, 0) if non_white_rows else 0
            bottom_trim = min(max(non_white_rows) + border, data.shape[0]) if non_white_rows else data.shape[0]

            cropped_img = img.crop((0, top_trim, data.shape[1], bottom_trim))
            cropped_img.save(os.path.join(self.out_directory, f"{filename}"))

    def delete_blank_images(self, image_path, tolerance=50):
        """
        Delete blank images from a directory.

        :param directory: Path to the directory containing images.
        :param tolerance: Tolerance value to consider a pixel as 'white' or close to white.
        """
        filename = image_path.split(os.path.sep)[-1]
        if is_image_blank(image_path, tolerance):
            os.remove(image_path)
            print(f"Deleted blank image: {filename}")




def merge_overlapping_boxes(boxes):
    """
    Merge overlapping boxes.

    :param boxes: List of bounding boxes (x1, y1, x2, y2).
    :return: List of merged bounding boxes.
    """
    merged = []
    while boxes:
        box = boxes.pop(0)
        overlaps = [b for b in boxes if do_overlap(box, b)]

        while overlaps:
            overlapping_box = overlaps.pop(0)
            boxes.remove(overlapping_box)
            box = combine_boxes(box, overlapping_box)
            overlaps = [b for b in boxes if do_overlap(box, b)]

        merged.append(box)
    return merged

def do_overlap(box1, box2):
    """
    Check if two boxes overlap.

    :param box1: First bounding box (x1, y1, x2, y2).
    :param box2: Second bounding box (x1, y1, x2, y2).
    :return: True if boxes overlap, False otherwise.
    """
    return not (box2[0] > box1[2] or box2[2] < box1[0] or box2[1] > box1[3] or box2[3] < box1[1])

def combine_boxes(box1, box2, enlarge_percent = 0.1, height_mult = 3, orig = None):
    """
    Combine two overlapping boxes into one.

    :param box1: First bounding box (x1, y1, x2, y2).
    :param box2: Second bounding box (x1, y1, x2, y2).
    :return: Combined bounding box (x1, y1, x2, y2).
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    # Calculate enlargement amount
    width_enlarge = (x2 - x1) * enlarge_percent 
    height_enlarge = (y2 - y1) * enlarge_percent * height_mult

    # Apply enlargement
    enlarged_box = (x1 - width_enlarge, y1 - height_enlarge, x2 + width_enlarge, y2 + height_enlarge)

    if orig is not None:
        # Ensure box coordinates do not exceed image boundaries
        img_width, img_height = orig
        enlarged_box = (
            max(0, min(enlarged_box[0], img_width)),  # x1
            max(0, min(enlarged_box[1], img_height)),  # y1
            max(0, min(enlarged_box[2], img_width)),  # x2
            max(0, min(enlarged_box[3], img_height))   # y2
        )

    return enlarged_box


def cluster_boxes(boxes, eps=120, min_samples=8):
    """
    Cluster boxes using DBSCAN based on their positions and area.

    :param boxes: List of bounding boxes (x1, y1, x2, y2).
    :param eps: Maximum distance between two boxes to be considered in the same neighborhood.
    :param min_samples: Minimum number of boxes in a neighborhood to form a cluster.
    :return: List of clustered bounding boxes.
    """
    # Calculate features: x_center, y_center, area
    features = []
    for box in boxes:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        area = (box[2] - box[0]) * (box[3] - box[1])
        features.append([x_center, y_center, area])
        #features.append([area])

    if features:
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
        labels = clustering.labels_

        # Group boxes by cluster
        clustered_boxes = {}
        for label, box in zip(labels, boxes):
            if label == -1:  # Skip noise
                continue
            if label not in clustered_boxes:
                clustered_boxes[label] = [box]
            else:
                clustered_boxes[label].append(box)

        # Combine boxes in each cluster
        combined_boxes = []
        for label, group in clustered_boxes.items():
            # Combine boxes in the cluster
            x1 = min(box[0] for box in group)
            y1 = min(box[1] for box in group)
            x2 = max(box[2] for box in group)
            y2 = max(box[3] for box in group)

            combined_boxes.append((x1, y1, x2, y2))

        # Merge overlapping boxes across clusters
        merged_boxes = merge_overlapping_boxes(combined_boxes)
    else:
        merged_boxes = []

    return merged_boxes

def process_region(image_path, region_box, eps=150, min_samples=10, enlarge_percent=0.1, lang='eng'):
    """
    Process a specific region of the image to refine bounding boxes.

    :param image: The original PIL image.
    :param region_box: The bounding box of the region to process (x1, y1, x2, y2).
    :param eps, min_samples, enlarge_percent: Parameters for box processing.
    :return: List of refined bounding boxes in the context of the original image.
    """
    image = Image.open(image_path)
    # Assuming 'image' is your PIL image object and 'region_box' is the box you want to crop
    img_width, img_height = image.size

    # Ensure the region box coordinates are within the image bounds
    x1, y1, x2, y2 = region_box
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    safe_region_box = (x1, y1, x2, y2)
    # Extract the region of interest with the adjusted box
    roi = image.crop(safe_region_box)
    img_width, img_height = roi.size
    # Convert to grayscale for OCR
    ocr_img = roi #.convert('L')
    boxes = pytesseract.image_to_boxes(ocr_img, lang=lang)

    # Process boxes
    processed_boxes = []
    for box in boxes.splitlines():
        char, left, bottom, right, top, _ = box.split(' ')
        left, bottom, right, top = map(int, [left, bottom, right, top])
        processed_boxes.append((left, roi.height - top, right, roi.height - bottom))

    # Cluster boxes
    clustered_boxes = cluster_boxes(processed_boxes, eps=eps, min_samples=min_samples)

    # Map boxes back to original image coordinates
    refined_boxes = []
    for box in clustered_boxes:
        enlarged_box = combine_boxes(box, box, enlarge_percent=enlarge_percent,  orig=(img_width, img_height))
        # Adjust coordinates to fit within the original image
        original_box = (enlarged_box[0] + region_box[0], enlarged_box[1] + region_box[1],
                        enlarged_box[2] + region_box[0], enlarged_box[3] + region_box[1])
        refined_boxes.append(original_box)

    # Convert box format
    formatted_boxes = []
    for box in refined_boxes:
        formatted_box = [(box[0], box[1]), (box[2], box[3])]
        formatted_boxes.append(formatted_box)

    return formatted_boxes

def draw_boxes_around_text(image_path, eps=150, min_samples=10, 
                           enlarge_percent=0.1, refine_percent=0.1,
                           save_image = False, lang='eng'):
    # Load the image
    img = Image.open(image_path)
    # Get the dimensions of the image
    img_width, img_height = img.size

    ocr_img = img.convert('L')
    boxes = pytesseract.image_to_boxes(ocr_img)

    # Process boxes
    processed_boxes = []
    for box in boxes.splitlines():
        char, left, bottom, right, top, _ = box.split(' ')
        left, bottom, right, top = map(int, [left, bottom, right, top])
        processed_boxes.append((left, img.height - top, right, img.height - bottom))
        #processed_boxes.append((left,  top, right,  bottom))

    # Cluster boxes
    clustered_boxes = cluster_boxes(processed_boxes, eps=eps, min_samples=min_samples)

    # Refine boxes
    out_refined_boxes = []
    for box in clustered_boxes:
        enlarged_box = combine_boxes(box, box, enlarge_percent=enlarge_percent, orig=(img_width, img_height))
        refined_boxes = process_region(image_path, enlarged_box, eps=eps, min_samples=min_samples, enlarge_percent=refine_percent, lang=lang)
        out_refined_boxes.extend(refined_boxes)

    # Draw refined boxes on the image
    draw = ImageDraw.Draw(img)
    for box in out_refined_boxes:
        draw.rectangle([box[0][0], box[0][1], box[1][0], box[1][1]], outline="red")

    out_refined_boxes = [[(min(box[0][0], box[1][0]), min(box[0][1], box[1][1]))
                          , (max(box[0][0], box[1][0]), max(box[0][1], box[1][1]))] for box in out_refined_boxes]

    # Save the image
    if save_image:
        img.save('debug_test.jpeg')

    return out_refined_boxes


def format_box_for_saving(boxes):
    """Format a single box from (x1, y1, x2, y2) to [[x1, y1], [x2, y2]]."""
    # adjusted_coords = []
    # if boxes:
    #     for box in boxes:
    #         x1,x2,y1,y2 = box
    #         #print(x1,x2,y1,y2)
    #         adjusted_coords.append([(x1,y1),(x2,y2)])
    #         #print(adjusted_coords)
    print('boxes in : ', boxes)
    return [[(y[0],y[1]),(y[2],y[3])] for y in boxes]
    #return adjusted_coords

def process_directory(input_dir = 'scratch', output_dir = 'scratch2', eps=150, 
                      min_samples=10, enlarge_percent=0.5, refine_percent=0.1,
                      lang='eng'):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            output_text_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")

            # Process the image
            refined_boxes = draw_boxes_around_text(image_path,  eps, min_samples, enlarge_percent, refine_percent, lang=lang)
            refined_boxes = [ x for x in refined_boxes if x != []]

            print('check refined boxes : ', refined_boxes)

            # Format and save boxes to .txt file
            formatted_boxes = refined_boxes #format_box_for_saving(refined_boxes)  #for box in refined_boxes]
            #with open(os.path.join(self.path['box_coords'], f'{image_name}.txt'), 'w') as f:
            #    json.dump(self.boxes, f)
            with open(output_text_path, 'w') as file:
                file.write(json.dumps(formatted_boxes))


def draw_all_ocr_boxes(image_path, output_path, lang = 'eng'):
    # Load the image
    img = Image.open(image_path)

    custom_config = r'--oem 1 --psm 13'
    # Perform OCR to get bounding boxes of text
    boxes = pytesseract.image_to_boxes(img, lang = lang) #, config=custom_config)

    # Draw each box on the image
    draw = ImageDraw.Draw(img)
    for box in boxes.splitlines():
        char, left, bottom, right, top, _ = box.split(' ')
        left, bottom, right, top = map(int, [left, bottom, right, top])
        
        # Adjust the box coordinates
        adjusted_box = (left, img.height - top, right, img.height - bottom)
        
        # Convert to the same structure as the 'good' box code
        start_x, end_x = sorted([adjusted_box[0], adjusted_box[2]])
        start_y, end_y = sorted([adjusted_box[1], adjusted_box[3]])

        # Now the box is in the format [(start_x, start_y), (end_x, end_y)]
        formatted_box = [(start_x, start_y), (end_x, end_y)]

        # If you need to draw the rectangle (similar to the 'good' code)
        # Assuming 'draw' is an instance of ImageDraw
        draw.rectangle([start_x, start_y, end_x, end_y], outline="red")

        # If you need to save/store the box
        # Assuming 'boxes' is a list where you store these
        boxes.append(formatted_box)

    # Save the image with boxes
    img.save(output_path)

# Example usage
# process_directory()
# filename = "scratch/page_num_35.jpeg"
# #filename = 'raw_chapters/english199/ocr_image/page_num_24.jpeg'
# draw_boxes_around_text(filename,enlarge_percent=0.5, refine_percent=0.1)
# pi(filename='test.jpg', height=600, width=200)

