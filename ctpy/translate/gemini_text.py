import requests
import os
import pyperclip
import json
import google.generativeai as genai
import re

from ..config import config
from .gpt_text import replace_page_references

PATHS = config()

with open(PATHS.text_refine, 'r', encoding='utf-8') as file:
    REFINE = file.read()



def translate_text_gemini(self, message):
    '''
    Function to translate OCR text using gemini
    '''
    genai.configure(api_key=self.gemini_key)
    model = genai.GenerativeModel('gemini-pro')

    payload = {
        "max_output_tokens": 4000,
        "temperature": 0.5,
        #"top_p": 1.0,
        #"top_k": 32, 
    }

    response = model.generate_content(message, generation_config=payload)


    txt_name = f'RAW{self.name}.txt'
    with open(os.path.join(self.path['prompt'],txt_name), 'w') as f:
        f.write(message)
    with open(os.path.join(self.path['response'],txt_name), 'w') as f:
        f.write(str(vars(response)))

    text = response.text
    with open(os.path.join(self.path['text'], txt_name), 'w') as file:
        file.write(text)

    return text

def refine_translation_gemini(self):
    '''
    Another shot at gpt to refine the translations
    '''
    txt_name = f'RAW{self.name}.txt'
    with open(os.path.join(self.path['text'], txt_name), 'r') as file:
        text = file.read()

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {self.api_key}"
    }

    message = REFINE + text

    genai.configure(api_key=self.gemini_key)
    model = genai.GenerativeModel('gemini-pro')

    payload = {
        "max_output_tokens": 4000,
        "temperature": 0.5,
        #"top_p": 1.0,
        #"top_k": 32, 
    }

    response = model.generate_content(message, generation_config=payload)

    txt_name = f'{self.name}.txt'
    with open(os.path.join(self.path['prompt'],txt_name), 'w') as f:
        f.write(message)
    with open(os.path.join(self.path['response'],txt_name), 'w') as f:
        f.write(str(vars(response)))

    text = response.text
    with open(os.path.join(self.path['text'], txt_name), 'w') as file:
        file.write(text)