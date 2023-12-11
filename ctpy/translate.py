import base64
import requests
import os
import json


from .config import config

PATHS = config()

with open(PATHS.prompt_path, 'r') as file:
    TEXT = file.read()
with open(PATHS.template_path, 'r') as file:
    TEMPLATE = file.read()
with open(PATHS.examples_path, 'r', encoding='utf-8') as file:
    EXAMPLES = file.read()
with open(PATHS.suffix_path, 'r') as file:
    SUFFIX = file.read()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

def translate_image(self, name, ocr_result, suffix = None):
  '''
  Function to translate an image
  '''

  path = os.path.join(self.path['image'], name)
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

  return response

def translate(self, debug = False):
    
    if self.force_debug:
        debug = True

    prefix = 'This is the response template you supplied for the previous page, to give you additional context.\n'
    suffix = SUFFIX
    prev_text = None
    for name in os.listdir(self.path['image']):
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
              response = self.translate_image(name, ocr_result, suffix = prev_text)
            try:
                text = response.json()['choices'][0]['message']['content']
                with open(os.path.join(self.path['text'], txt_name), 'w') as file:
                    file.write(text)
            except:
               print('GPT failure on ', name)
               with open(os.path.join(self.path['text'], txt_name), 'w') as file:
                    text = 'IMAGE_DESCRIPTION\nGPT failure on this image\nBEGIN_TEXT\nGPT failure\nEND_TEXT\n'
                    file.write(text)
                    
        else:
            with open(os.path.join(self.path['text'], txt_name), 'r', encoding='utf-8') as file:
                text = file.read()
        prev_text = prefix + text + suffix