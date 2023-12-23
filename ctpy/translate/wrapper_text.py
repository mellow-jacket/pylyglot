
import os
import google.generativeai as genai
from ..config import config

from ..splits import natural_sort_key
from .gpt_text import _translate_text, refine_translation
from .gemini_text import translate_text_gemini, refine_translation_gemini

PATHS = config()

with open(PATHS.text_prompt, 'r') as file:
    TEXT = file.read()
with open(PATHS.text_examples, 'r', encoding='utf-8') as file:
    EXAMPLES = file.read()


def translate_text(self, model = 'gpt'):
    message = TEXT + EXAMPLES
    message+= '\nHere is the korean text. Please remember to only respond with only the translated english, and indicated page number.\n'
    all_images = sorted(os.listdir(self.path['ocr_text']), key=natural_sort_key)
    for name in all_images:
        path = os.path.join(self.path['ocr_text'], name)
        with open(path, 'r') as f:
            message+=f.read()
    message+='\n Please pay close attention to all of my instructions. Thank you for your help!'
    if model == 'gpt':
        _translate_text(self, message)
        refine_translation(self)
    elif model == 'gemini':
        translate_text_gemini(self,message)
        refine_translation_gemini(self)