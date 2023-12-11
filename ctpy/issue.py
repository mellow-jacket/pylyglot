
import os

from PIL import Image, ImageDraw
import pytesseract
import shutil
import json
from .config import config
from .img_tools import downscale_image, _combine_pages
from .translate import translate, translate_image
from .make_pdf import make_pdf
from .img_download import scrape

IMGPATH = config().img_path
TESTPATH = config().testimg_path

class issue:

    def __init__(self,
                name = 'test158',
                url = 'https://newtoki317.com/webtoon/32675929?toon=%EC%9D%BC%EB%B0%98%EC%9B%B9%ED%88%B0',
                api_key = None,
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

        self.raw_direc = 'raw_chapters'
        self.final_direc = 'final_chapters'

        base_path = os.path.join(IMGPATH,self.raw_direc, name)
        self.path = {
            'base':base_path,
            'image':os.path.join(base_path, 'images'),
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
        for key, dir_path in self.path.items():
            # Skip the 'base' and 'raw_image' directories
            if key in ['base', 'raw_image']:
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

    def translate(self, debug = False):
        translate(self, debug = debug)

    def translate_image(self, name, ocr_result, suffix = None):
        return translate_image(self, name, ocr_result, suffix = suffix)

    def make_pdf(self):
        make_pdf(self)

    def scrape(self):
        if self.url is None:
            for filename in os.listdir(TESTPATH):
                file_path = os.path.join(TESTPATH, filename)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, self.path['raw_image'])        
        else:
            scrape(self)

    def downsample(self, scale_factor = 0.5):
        for file in os.listdir(self.path['raw_image']):
            path = os.path.join(self.path['raw_image'],file)
            image, base, ocr_img = downscale_image(path, scale_factor=scale_factor)
            image.save(os.path.join(self.path['image'],file))
            base.save(os.path.join(self.path['raw_combined'],file))
            ocr_img.save(os.path.join(self.path['ocr_image'],file))

    def combine_pages(self,*pages, name = None):
        for src in ['image', 'raw_combined', 'ocr_image']:
            _combine_pages(self, *pages, name = name, which = src)

    def ocr(self, lang = 'kor'):
        '''
        Pull possible korean text from processed image
        '''
        paths = [ x for x in os.listdir(self.path['ocr_image'])]
        for name in paths:
            img_path = os.path.join(self.path['ocr_image'], name)
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img, lang=lang, config=f'--oem 3 --psm 11') #psm 3 or 11
            text = text.replace('\n','')
            text = text.replace('PSM 11 Result:','')
            text = 'OCR analysis :'+text+'\n'
            with open(os.path.join(self.path['ocr_text'], name.split('.')[0]+'.txt'), 'w') as file:
                file.write(text)

    def combine_gpt_and_ocr(self):
        gpt_images = os.listdir(self.path['gpt_image'])
        for img_name in gpt_images:
            gpt_img_path = os.path.join(self.path['gpt_image'], img_name)
            ocr_img_path = os.path.join(self.path['ocr_image'], img_name)

            if os.path.exists(gpt_img_path) and os.path.exists(ocr_img_path):
                gpt_image = Image.open(gpt_img_path)
                ocr_image = Image.open(ocr_img_path)

                # Concatenate images horizontally
                total_width = gpt_image.width + ocr_image.width
                max_height = max(gpt_image.height, ocr_image.height)
                combined_img = Image.new('RGB', (total_width, max_height))
                combined_img.paste(gpt_image, (0, 0))
                combined_img.paste(ocr_image, (gpt_image.width, 0))

                # Save the combined image
                combined_img.save(gpt_img_path)

    def make_gpt_images(self):
        images = os.listdir(self.path['image'])
        for img_name in images:
            image_name = img_name.split('.')[0]
            boxes = []
            with open(os.path.join(self.path['box_coords'], f'{image_name}.txt'), 'w') as f:
                json.dump(boxes, f)

            # Save image with boxes
            image = Image.open(os.path.join(self.path['image'], f'{image_name}.jpeg'))
            ocr_image = Image.open(os.path.join(self.path['ocr_image'], f'{image_name}.jpeg'))
            if boxes:  # Check if there are boxes to draw
                draw = ImageDraw.Draw(image)
                draw2 = ImageDraw.Draw(ocr_image)
                for box in boxes:
                    draw.rectangle([box[0], box[1]], outline="red", width=5)
                    draw2.rectangle([box[0], box[1]], outline="red", width=5)
            image.save(os.path.join(self.path['gpt_image'], f'{image_name}.jpeg'))
            ocr_image.save(os.path.join(self.path['ocr_image'], f'{image_name}.jpeg'))

        self.combine_gpt_and_ocr()