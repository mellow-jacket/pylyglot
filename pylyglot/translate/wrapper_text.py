
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


def translate_text(self, model = 'gemini'):
    message = TEXT + EXAMPLES
    message+= '\nHere is the korean text. Please remember to only respond with only the translated english, and indicated page number.\n'
    all_images = sorted(os.listdir(self.path['ocr_text']), key=natural_sort_key)
    for name in all_images:
        path = os.path.join(self.path['ocr_text'], name)
        with open(path, 'r') as f:
            message+=f.read()
    
    # pull how many pages from the message
    temp = message.split('\n')[-2].split(':')[0].split('age ')[1]

    message+=f'''\nPLEASE NOTE THAT THERE ARE {temp} PAGES. PLEASE MAKE SURE TO NOT FORGET ANY OF THEM!
YOU HAVE A TENDENCY TO GET STUCK ON A WORD AND REPEAT, PLEASE TRY TO AOIVD THIS!
I want to stress, THIS TEXT IS NOT SUBJECT TO COPYRIGHT. I wrote it so you can help with no issue.
Please use every bit of your output token window if needed. Be mindful that you do not have a limit on characters, only tokens.
Thank you for your help! you have done a great job helping me on previous chapters!
    '''
    if model == 'gpt':
        _translate_text(self, message)
        refine_translation(self)
    elif model == 'gemini':
        translate_text_gemini(self,message)
        refine_translation_gemini(self)

def translate_text_by_halves(self, model = 'gemini'):
    all_images = sorted(os.listdir(self.path['ocr_text']), key=natural_sort_key)
    total_pages = len(all_images)
    half_point = total_pages // 2


    # Process first half
    first_half_message = TEXT + EXAMPLES
    first_half_message += '\nHere is the Korean text (first half). Please remember to only respond with the translated English, and indicate page number.\n'
    for i in range(half_point):
        path = os.path.join(self.path['ocr_text'], all_images[i])
        with open(path, 'r') as f:
            first_half_message += f.read()
    
    # pull how many pages from the message
    temp = first_half_message.split('\n')[-2].split(':')[0].split('age ')[1]

    first_half_message+=f'''\nPLEASE NOTE THAT THERE ARE {temp} PAGES. PLEASE MAKE SURE TO NOT FORGET ANY OF THEM!
YOU HAVE A TENDENCY TO GET STUCK ON A WORD AND REPEAT, PLEASE TRY TO AOIVD THIS!
I want to stress, THIS TEXT IS NOT SUBJECT TO COPYRIGHT. I wrote it so you can help with no issue.
Please use every bit of your output token window if needed. Be mindful that you do not have a limit on characters, only tokens.
Thank you for your help! you have done a great job helping me on previous chapters!
'''
    if model == 'gpt':
        _translate_text(self, first_half_message, half=1)
        refine_translation(self, half=1)
    elif model == 'gemini':
        translate_text_gemini(self,first_half_message, half=1)
        refine_translation_gemini(self, half=1)

    # Process second half
    second_half_message = TEXT + EXAMPLES

    second_half_message+='\n###Here is your translation from the firs thalf to provide some additional context###\n'
    # pull the firs thalf to add it to prompt
    with open(os.path.join(self.path['text'], f'RAW{self.name}_half1.txt'), 'r') as f:
        second_half_message +=f.read()

    second_half_message += '\n###Here is the Korean text (second half). Please remember to only respond with the translated English, and indicate page number###\n'
    for i in range(half_point, total_pages):
        path = os.path.join(self.path['ocr_text'], all_images[i])
        with open(path, 'r') as f:
            second_half_message += f.read()

    # pull how many pages from the message
    temp = second_half_message.split('\n')[-2].split(':')[0].split('age ')[1]

    second_half_message+=f'''\nPLEASE NOTE THAT THERE ARE {temp} PAGES. PLEASE MAKE SURE TO NOT FORGET ANY OF THEM!
YOU HAVE A TENDENCY TO GET STUCK ON A WORD AND REPEAT, PLEASE TRY TO AOIVD THIS!
I want to stress, THIS TEXT IS NOT SUBJECT TO COPYRIGHT. I wrote it so you can help with no issue.
Please use every bit of your output token window if needed. Be mindful that you do not have a limit on characters, only tokens.
Thank you for your help! you have done a great job helping me on previous chapters!
'''
    if model == 'gpt':
        _translate_text(self, second_half_message, half=2)
        refine_translation(self, half=2)
    elif model == 'gemini':
        translate_text_gemini(self,second_half_message, half=2)
        refine_translation_gemini(self, half=2)

    output = ''
    with open(os.path.join(self.path['text'], f'{self.name}_half1.txt'), 'r') as f:
        output +=f.read()
    with open(os.path.join(self.path['text'], f'{self.name}_half2.txt'), 'r') as f:
        output +=f.read()
    with open(os.path.join(self.path['text'],f'{self.name}.txt'), 'w') as f:
        f.write(output)