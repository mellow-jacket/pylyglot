import requests
import os
import pyperclip
import json
import google.generativeai as genai
import re

from ..config import config


PATHS = config()

with open(PATHS.text_refine, 'r', encoding='utf-8') as file:
    REFINE = file.read()


def replace_page_references(text):
    # This regex pattern matches "Page " followed by 1 to 4 digits
    pattern = r"Page \d{1,4}:"
    
    # Replace matched patterns with "[Page Reference]"
    replaced_text = re.sub(pattern, "", text)
    
    return replaced_text

def _translate_text(self,  message):
    '''
    Function to translate an image
    '''


    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {self.api_key}"
    }


    payload = {
    "model": "gpt-4-1106-preview",  # or whichever GPT model you are using
    "messages": [
        {
            "role": "user",
            "content": message
        }
    ],
    "max_tokens": 4096
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    txt_name = f'RAW{self.name}.txt'
    with open(os.path.join(self.path['prompt'],txt_name), 'w') as f:
        f.write(message)
    with open(os.path.join(self.path['response'],txt_name), 'w', encoding='utf-8') as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=4)

    #try:
    text = response.json()['choices'][0]['message']['content']
    with open(os.path.join(self.path['text'], txt_name), 'w') as file:
        file.write(text)
    # except:
    #     print('GPT failure on ', self.name)
    #     with open(os.path.join(self.path['text'], txt_name), 'w') as file:
    #         text = str(response.json())
    #         file.write(text)

def refine_translation(self):
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

    payload = {
    "model": "gpt-4-1106-preview",  # or whichever GPT model you are using
    "messages": [
        {
            "role": "user",
            "content": message
        }
    ],
    "max_tokens": 4096
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    txt_name = f'{self.name}.txt'
    with open(os.path.join(self.path['prompt'],txt_name), 'w') as f:
        f.write(message)
    with open(os.path.join(self.path['response'],txt_name), 'w', encoding='utf-8') as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=4)

    text = response.json()['choices'][0]['message']['content']
    #outtext = replace_page_references(text)
    with open(os.path.join(self.path['text'], txt_name), 'w') as file:
        file.write(text)


def load_chapter(self):
    # Load the file content into a string
    with open(os.path.join(self.path['text'], f'{self.name}.txt'), 'r') as file:
        text = file.read()

    # Process the text
    processed_text = replace_page_references(text)

    # Copy the processed text to the clipboard
    pyperclip.copy(processed_text)