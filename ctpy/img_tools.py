from PIL import Image
from PIL import Image, ImageFilter, ImageEnhance
import os
import shutil

def get_size(image_path):
    """
    Function to get the dimensions of an image and calculate the total number of pixels.
    
    :param image_path: Path to the image file
    :return: tuple (width, height, total_pixels)
    """
    with Image.open(image_path) as img:
        width, height = img.size
        total_pixels = width * height
        return width, height, total_pixels
    

def downscale_image(image_path, scale_factor = 3.0):
    """
    Function to downscale an image by a given scale factor.

    :param image_path: Path to the image file
    :param scale_factor: Factor by which to scale down the image (e.g., 0.5 to reduce size by half)
    :return: Downscaled image
    """
    with Image.open(image_path) as img:
        # Calculate new dimensions
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        base_image = img
        # Resize the image using the LANCZOS resampling filter
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Convert to grayscale
        ocr_img = resized_img.convert('L')

        # Increase contrast
        enhancer = ImageEnhance.Contrast(ocr_img)
        ocr_img = enhancer.enhance(1.5)  # play around with the factor as needed

        # Binarize the image
        threshold = 190  # This is an example threshold value
        ocr_img = ocr_img.point(lambda p: p > threshold and 255)

        # Denoise
        ocr_img = ocr_img.filter(ImageFilter.MedianFilter(size=3))

        # Sharpen
        ocr_img = ocr_img.filter(ImageFilter.UnsharpMask())

        # Return the resized image
    return resized_img, base_image, ocr_img
    
def _combine_pages(self,*pages, name = None, which='image'):
    """
    Function to combine multiple images (pages) vertically in the order they are provided.
    The images can be either paths or base64 encoded strings, depending on base64_input flag.

    :param pages: Paths to the images or base64 encoded strings of the images
    :param base64_input: Flag indicating whether the pages are base64 encoded strings
    :return: Combined image
    """
    if name is None:
        new_name = 'page_num'
        for i in pages:
            new_name+=f'_{i}'
    else:
        new_name = name

    paths = [os.path.join(self.path[which],f'page_num_{x}.jpeg') for x in pages]
    images = [ Image.open(x) for x in paths]

    # Determine the total height of the combined image
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    # Create a new image with the appropriate combined size
    combined_image = Image.new('RGB', (max_width, total_height))

    # Paste each image into the combined image
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    combined_image.save(os.path.join(self.path[which],new_name)+'.jpeg')
    for x in paths:
        os.remove(x)

def copy_files(source_dir, dest_dir):
    # Create the destination directory if it does not exist
    if os.path.exists(dest_dir):
        for filename in os.listdir(dest_dir):
            file_path = os.path.join(dest_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        # Create the destination directory if it does not exist
        os.makedirs(dest_dir)

    # Copy each file from the source directory to the destination directory
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest_dir)