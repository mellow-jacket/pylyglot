'''
Submodule for all api calls
'''

from .gpt_vision import translate_image_gpt
from .gemini_vision import translate_image_gemini
from .gpt_text import load_chapter
from .wrapper_text import translate_text, translate_text_by_halves
from .wrapper_vision import translate_images