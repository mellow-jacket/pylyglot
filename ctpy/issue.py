
import os

from PIL import Image, ImageDraw
import pytesseract
import shutil
import json

from .config import config
from .img_tools import downscale_image, _combine_pages
from .translate import translate, translate_image_gpt, translate_image_gemini
from .make_pdf import make_pdf
from .img_download import scrape, scrape_md
from .ocr_tools import perform_ocr_on_all_images
from .autobox import md_image_cleaner, process_directory
from .splits import increment_file_names, reindex_file_names


cfg = config()
IMGPATH = cfg.img_path
TESTPATH = cfg.testimg_path
BENCHMARK = cfg.benchmark_path

class issue:
    '''
    Class for managing translation of a chapter
    Many things are stored in a directory structed found in...

        self.raw_direc/self.name
        raw_chapters/test158 - by default args

    beware of text boxes, they are still very manual and not saved well
    '''
    def __init__(self,
                name = 'test158',
                url = 'https://newtoki317.com/webtoon/32675929?toon=%EC%9D%BC%EB%B0%98%EC%9B%B9%ED%88%B0',
                api_key = None,
                gemini_key = None,
                ):
        '''
        Issue of skeleton soldier
        '''
        self.name = name
        self.url = url

        if api_key is None:
            self.force_debug = True
            print('Debug mode is forced with no api key')
        else:
            self.force_debug = False
        self.api_key = api_key

        self.gemini_key = gemini_key

        self.raw_direc = 'raw_chapters'
        self.final_direc = 'final_chapters'

        base_path = os.path.join(IMGPATH,self.raw_direc, name)
        self.path = {
            'base':base_path,
            'image':os.path.join(base_path, 'images'),
            'scraped_image':os.path.join(base_path, 'scraped_images'),
            'raw_image':os.path.join(base_path, 'raw_images'),
            'text':os.path.join(base_path, 'text'),
            'raw_combined':os.path.join(base_path, 'raw_combined'),
            'ocr_image':os.path.join(base_path, 'ocr_image'),
            'ocr_text':os.path.join(base_path, 'ocr_text'),
            'gpt_image':os.path.join(base_path, 'gpt_image'),
            'box_coords':os.path.join(base_path, 'box_coords'),
            'prompt':os.path.join(base_path, 'prompt'),
            'response':os.path.join(base_path, 'response'),
                    }
        for key in self.path:
            direc = self.path[key]
            if not os.path.exists(direc):
                os.makedirs(direc)

    def clear_directories(self):
        '''
        housekeeping function to clear all directoires EXCEPT raw_images
        '''
        for key, dir_path in self.path.items():
            # Skip the 'base' and 'raw_image' directories
            if key in ['base', 'raw_image', 'scraped_image']:
                continue

            # Check if the directory exists
            if os.path.exists(dir_path):
                # Delete all files in the directory
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

                # Optionally, recreate the directory if needed
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

    def translate(self, debug = False, model = 'gpt'):
        '''
        Wrapper for translate.py/translate
        '''
        translate(self, debug = debug, model = model)

    def translate_image_gpt(self, name, ocr_result, suffix = None):
        '''
        Wrapper for translate.py/translate_image
        '''
        return translate_image_gpt(self, name, ocr_result, suffix = suffix)

    def translate_image_gemini(self, name, ocr_result, suffix = None):
        '''
        Wrapper for translate.py/translate_image
        '''
        return translate_image_gemini(self, name, ocr_result, suffix = suffix)

    def autobox(self, lang='kor'):
        '''
        Automatically generate some boxes on images
        '''
        process_directory(input_dir=self.path['image'], output_dir=self.path['box_coords'], lang=lang)

    def make_pdf(self):
        '''
        Wrapper for make_pdf.py/make_pdf
        '''
        make_pdf(self)

    def md_scrape(self):
        scrape_md(self)
        self.reindex_direc()

    def md_autosplit(self):
        cleaner = md_image_cleaner(in_directory=self.path['scraped_image'], out_directory = self.path['raw_image'])
        cleaner.split_raws()
        cleaner.clean_splits()
        self.reindex_direc()

    def reindex_direc(self, label = 'raw_image'):
            # reindex file names given label for path 
            reindex_file_names(self.path[label])

    def scrape(self):
        '''
        wrapper for img_download.py/scrap
        Exceptions handled here
        '''
        if self.url is None:
            for filename in os.listdir(TESTPATH):
                file_path = os.path.join(TESTPATH, filename)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, self.path['scraped_image'])        
        elif self.url == 'benchmark':
            for filename in os.listdir(BENCHMARK):
                file_path = os.path.join(BENCHMARK, filename)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, self.path['scraped_image'])  
        else:
            scrape(self)

        # populate raw from scraped
        if not os.listdir(self.path['raw_image']):
            for filename in os.listdir(self.path['scraped_image']):
                file_path = os.path.join(self.path['scraped_image'], filename)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, self.path['raw_image'])  



    def downsample(self, scale_factor = 0.5):
        '''
        down(re) sample image to different size
        '''
        for file in os.listdir(self.path['raw_image']):
            path = os.path.join(self.path['raw_image'],file)
            image, base, ocr_img = downscale_image(path, scale_factor=scale_factor)
            image.save(os.path.join(self.path['image'],file))
            base.save(os.path.join(self.path['raw_combined'],file))
            ocr_img.save(os.path.join(self.path['ocr_image'],file))

    def combine_pages(self,*pages, name = None):
        '''
        Combine pages in ['image', 'raw_combined', 'ocr_image']
        '''
        for src in ['image', 'raw_combined', 'ocr_image']:
            _combine_pages(self, *pages, name = name, which = src)

    def ocr(self, lang = 'kor'):
        '''
        Wrapper for ocr_tools.py/ocr
        '''
        perform_ocr_on_all_images(self, lang=lang)

    def perform_ocr_on_all_images(self, lang = 'kor'):
        '''
        keeping this because I will forget
        '''
        perform_ocr_on_all_images(self, lang=lang)

    def copy_to_gpt_images(self):
        src_directory = self.path['image']
        dest_directory = self.path['gpt_image']
        # Copy each file from the source directory to the destination directory
        for filename in os.listdir(src_directory):
            file_path = os.path.join(src_directory, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, dest_directory)

    def combine_gpt_and_ocr(self):
        '''
        LIKELY DEPRECATED
        Add black&white processed OCR image to the gpt_image
        '''
        self.copy_to_gpt_images()
        gpt_images = os.listdir(self.path['image'])
        for img_name in gpt_images:
            gpt_img_path = os.path.join(self.path['gpt_image'], img_name)
            img_path = os.path.join(self.path['image'], img_name)
            ocr_img_path = os.path.join(self.path['ocr_image'], img_name)

            if os.path.exists(img_path) and os.path.exists(ocr_img_path):
                img_image = Image.open(img_path)
                ocr_image = Image.open(ocr_img_path)

                # Concatenate images horizontally
                total_width = img_image.width + ocr_image.width
                max_height = max(img_image.height, ocr_image.height)
                combined_img = Image.new('RGB', (total_width, max_height))
                combined_img.paste(img_image, (0, 0))
                combined_img.paste(ocr_image, (img_image.width, 0))

                # Save the combined image
                combined_img.save(gpt_img_path)

    def add_boxes_to_images(self):
        images = os.listdir(self.path['image'])
        for img_name in images:
            image_name = img_name.split('.')[0]
            # Load the box coordinates
            boxes_path = os.path.join(self.path['box_coords'], f'{image_name}.txt')

            with open(boxes_path, 'r') as f:
                boxes = json.load(f)

            # Load and draw boxes on the image
            image_path = os.path.join(self.path['image'], img_name)
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            for box in boxes:
                # Flatten the box list and convert to integers
                flat_box = [int(coord) for point in box for coord in point]
                draw.rectangle(flat_box, outline="red", width=5)

            # Load and draw boxes on the ocr image
            ocr_path = os.path.join(self.path['ocr_image'], img_name)
            ocr_image = Image.open(ocr_path)
            ocr_draw = ImageDraw.Draw(ocr_image)
            for box in boxes:
                # Flatten the box list and convert to integers
                flat_box = [int(coord) for point in box for coord in point]
                ocr_draw.rectangle(flat_box, outline="red", width=5)

            # Save the image with boxes
            image.save(os.path.join(self.path['image'], img_name))
            ocr_image.save(os.path.join(self.path['ocr_image'], img_name))

    def tile_page_for_gpt(self):
        '''
        Tile the base page and text boxes for more dense information
        '''
        self.copy_to_gpt_images()
        gpt_images = os.listdir(self.path['image'])
        for img_name in gpt_images:
            self.create_composite_image(img_name.split('.')[0])

    def create_composite_image(self, image_name, max_pixels=768, add_original=True, border_size=2):
        # Paths
        image_path = os.path.join(self.path['gpt_image'], f'{image_name}.jpeg')
        boxes_path = os.path.join(self.path['box_coords'], f'{image_name}.txt')

        # Load the base image
        base_image = Image.open(image_path)
        
        remaining_height = max_pixels

        # Rotate and scale the base image if needed
        if add_original:
            rotated_image = base_image.rotate(90, expand=True)
            scale_factor = max_pixels / rotated_image.width
            scaled_rotated_image = rotated_image.resize((max_pixels, int(rotated_image.height * scale_factor)), Image.Resampling.LANCZOS)
            remaining_height -= scaled_rotated_image.height + border_size

        # Load boxes and prepare text images
        with open(boxes_path, 'r') as f:
            boxes = json.load(f)

        columns = max(min(len(boxes), 3),1)  # Max of 3 columns
        column_width = (max_pixels - (border_size * (columns + 1))) // columns

        text_images = [self._resize_text_image(base_image.crop((box[0][0], box[0][1], box[1][0], box[1][1])), column_width, remaining_height) for box in boxes]

        # Create a new composite image
        new_image = Image.new('RGB', (max_pixels, max_pixels))  # Initial large size
        y_offset = self._paste_rotated_image(new_image, scaled_rotated_image, border_size) if add_original else border_size

        # Paste text images and get the max y-coordinate used
        max_y_coordinate = self._paste_text_images(new_image, text_images, max_pixels, border_size, y_offset, remaining_height)
        
        # Crop the image to remove any unused space at the bottom
        final_image = new_image.crop((0, 0, max_pixels, max_y_coordinate))
        
        # Save the final cropped image
        final_image.save(image_path)

    def _resize_text_image(self, img, max_width, max_height):
        aspect_ratio = img.height / img.width
        new_width = min(max_width, int(max_height / aspect_ratio))
        new_height = int(new_width * aspect_ratio)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _paste_rotated_image(self, new_image, rotated_image, border_size):
        new_image.paste(rotated_image, (border_size, border_size))
        return rotated_image.height + border_size

    def _paste_text_images(self, new_image, text_images, max_pixels, border_size, y_offset, remaining_height):
        max_y_coordinate = max_pixels - remaining_height
        if len(text_images) == 1:
            # If there's only one text image, use the remaining space for it
            text_image = text_images[0]
            resized_text_image = self._resize_text_image(text_image, max_pixels - 2 * border_size, remaining_height)
            new_image.paste(resized_text_image, (border_size, y_offset + border_size))
            max_y_coordinate = max(max_y_coordinate, y_offset + resized_text_image.height)
        else:
            x_offset, row_height = border_size, 0
            for img in text_images:
                if x_offset + img.width + border_size > max_pixels:
                    x_offset = border_size
                    y_offset += row_height + border_size
                    row_height = img.height
                new_image.paste(img, (x_offset, y_offset))
                x_offset += img.width + border_size
                row_height = max(row_height, img.height)
                max_y_coordinate = max(max_y_coordinate, y_offset + img.height)
        return max_y_coordinate + border_size
    

    def resize_gpt(self, scale=2.0):
        image_dir = self.path['gpt_image']
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            with Image.open(file_path) as img:
                new_size = tuple([int(dim * scale) for dim in img.size])
                img = img.resize(new_size, Image.LANCZOS)
            img.save(file_path)