import base64
import requests
import os
import json
import PIL.Image as Image
import google.generativeai as genai
from ..config import config

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

def translate_image_gpt(self, name, ocr_result, suffix = None):
  '''
  Function to translate an image
  '''

  path = os.path.join(self.path[IMAGE_PATH], name)
  # Getting the base64 string
  base64_image = encode_image(path) 

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {self.api_key}"
  }

  message = TEXT + ocr_result + TEMPLATE + EXAMPLES
  if suffix is not None:
      message+=suffix

  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            'type':'image_url',
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          },
          {
            "type": "text",
            "text": message
          }
        ]
      }
    ],
    "max_tokens": 500
  }
  
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  
  txt_name = f'{name.split(".")[0]}.txt'
  with open(os.path.join(self.path['prompt'],txt_name), 'w') as f:
      f.write(message)
  with open(os.path.join(self.path['response'],txt_name), 'w', encoding='utf-8') as f:
    json.dump(response.json(), f, ensure_ascii=False, indent=4)

  try:
      text = response.json()['choices'][0]['message']['content']
      with open(os.path.join(self.path['text'], txt_name), 'w') as file:
          file.write(text)
  except:
      print('GPT failure on ', name)
      with open(os.path.join(self.path['text'], txt_name), 'w') as file:
          text = 'IMAGE_DESCRIPTION\nGPT failure on this image\nBEGIN_TEXT\nGPT failure\nEND_TEXT\n'
          file.write(text)

  return text