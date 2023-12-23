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