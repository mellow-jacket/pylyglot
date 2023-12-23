import base64
import requests
import os
import json
import PIL.Image as Image
import google.generativeai as genai
from ..config import config

from .gpt_vision import translate_image_gpt
from .gemini_vision import translate_image_gemini

from ..splits import natural_sort_key

PATHS = config()

IMAGE_PATH = 'gpt_image'

# with open(PATHS.gpt_prompt, 'r') as file:
#     TEXT = file.read()
with open(PATHS.box_prompt, 'r') as file:
    TEXT = file.read()
with open(PATHS.gpt_template, 'r') as file:
    TEMPLATE = file.read()
with open(PATHS.gpt_examples, 'r', encoding='utf-8') as file:
    EXAMPLES = file.read()
with open(PATHS.gpt_suffix, 'r') as file:
    SUFFIX = file.read()

with open(PATHS.gemini_prompt, 'r') as file:
    GEM_TEXT = file.read()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')
    
def translate_images(self, debug = False, model = 'gpt'):
    
    if self.force_debug:
        debug = True

    prefix = 'This is the response template you supplied for the previous page, to give you additional context.\n'
    suffix = SUFFIX
    prev_text = None
    all_images = sorted(os.listdir(self.path[IMAGE_PATH]), key=natural_sort_key)
    for name in all_images:
        txt_name = name.replace('.jpeg','.txt')
        box_coords_path = os.path.join(self.path['box_coords'], txt_name)

        # Check if the box coordinates file is empty (only contains [])
        if os.path.exists(box_coords_path):
            with open(box_coords_path, 'r') as file:
                box_coords = json.load(file)
            if not box_coords:  # Empty list
              with open(os.path.join(self.path['text'], txt_name), 'w') as file:
                    text = 'IMAGE_DESCRIPTION\nImage skipped by user\nBEGIN_TEXT\nUser designated skip\nEND_TEXT\n'
                    file.write(text)
              continue  # Skip this file
            
        # skip any .txt files that exists
        if not os.path.exists(os.path.join(self.path['text'], txt_name)):
            
            with open(os.path.join(self.path['ocr_text'], txt_name), 'r') as file:
                ocr_result = file.read()
            if not debug:
              if model == 'gpt':
                text = translate_image_gpt(self, name, ocr_result, suffix = prev_text)
              elif model == 'gemini':
                text = translate_image_gemini(self, name, ocr_result, suffix = prev_text)
              else:
                  raise ValueError('must specify model = gpt or gemini')

        else:
            with open(os.path.join(self.path['text'], txt_name), 'r', encoding='utf-8') as file:
                text = file.read()
        prev_text = prefix + text + suffix