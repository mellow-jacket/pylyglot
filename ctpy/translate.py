import base64
import requests
import os
import json
import PIL.Image as Image
import google.generativeai as genai
from .config import config

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

def translate_image_gemini(self, name, ocr_result, suffix = None):
  '''
  Function to translate an image
  '''

  path = os.path.join(self.path[IMAGE_PATH], name)
  # Getting the base64 string
  image = Image.open(path)
  genai.configure(api_key=self.gemini_key)

  message = GEM_TEXT + ocr_result + TEMPLATE # + EXAMPLES
  if suffix is not None:
      message+=suffix

  model = genai.GenerativeModel('gemini-pro-vision')
  
  payload = {
        "max_output_tokens": 2048,
        "temperature": 0.5,
        #"top_p": 1.0,
        #"top_k": 32, 
  }
  
  response = model.generate_content([image, message], generation_config=payload)


  txt_name = f'{name.split(".")[0]}.txt'
  with open(os.path.join(self.path['prompt'],txt_name), 'w') as f:
      f.write(message)
  with open(os.path.join(self.path['response'],txt_name), 'w') as f:
    f.write(str(vars(response)))

  try:
      text = response.text
      with open(os.path.join(self.path['text'], txt_name), 'w') as file:
          file.write(text)
  except:
      print('Gemini failure on ', name)
      with open(os.path.join(self.path['text'], txt_name), 'w') as file:
          text = 'IMAGE_DESCRIPTION\nGemini failure on this image\nBEGIN_TEXT\nGemini failure\nEND_TEXT\n'
          file.write(text)

  return text
def translate(self, debug = False, model = 'gpt'):
    
    if self.force_debug:
        debug = True

    prefix = 'This is the response template you supplied for the previous page, to give you additional context.\n'
    suffix = SUFFIX
    prev_text = None
    for name in os.listdir(self.path[IMAGE_PATH]):
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
                text = self.translate_image_gpt(name, ocr_result, suffix = prev_text)
              elif model == 'gemini':
                text = self.translate_image_gemini(name, ocr_result, suffix = prev_text)
              else:
                  raise ValueError('must specify model = gpt or gemini')

        else:
            with open(os.path.join(self.path['text'], txt_name), 'r', encoding='utf-8') as file:
                text = file.read()
        prev_text = prefix + text + suffix

