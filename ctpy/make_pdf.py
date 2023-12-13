from PIL import Image, ImageDraw, ImageFont
import textwrap
from reportlab.pdfgen import canvas
import uuid
import os
import re


BASE_FONT_SIZE = 25
BASE_IMAGE_HEIGHT = 1100
LINEWIDTH = 18
BASE_LINE_HEIGHT = 25


def contains_korean(text):
    korean_regex = r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uD7B0-\uD7FF]+'
    return bool(re.search(korean_regex, text))

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def scale_font_line_height(image_height, base_height=BASE_IMAGE_HEIGHT, base_font_size=BASE_FONT_SIZE, base_line_height=BASE_LINE_HEIGHT):
    """
    Scale the font size and line height based on the height of the image.
    """
    scale_factor = max(2,image_height / base_height)
    return int(scale_factor * base_font_size), int(scale_factor * base_line_height), 

def wrap_text(text, max_line_length, font, draw):
    skip_list = ['<Korean text>', '<English text>', '<Hangul text>', '<English translation>',
                 'English text', 'Korean text', '<sound effect>', 'KOREAN_TEXT', 'ENGLISH_TEXT',
                 '<text>', 'END_TRANSLATION', '(no spoken text', '(No English','(No additional English',
                 '! (This symbol repre', 'no text', 'No text', 'END_TEXT','User designated skip', #'Please note',
                 ]
    wrapped_lines = []
    consecutive_breaks = 0
    korean_counter = 0
    english_counter = 0

    for line in text.split('\n'):
        if 'KOREAN_TEXT' in line:
            korean_counter += 1
        if 'ENGLISH_TEXT' in line:
            english_counter += 1

        # Check if both Korean and English sections have appeared
        if korean_counter > 1 or english_counter > 1:
            wrapped_lines.append('')  # Add a line break after both sections
            korean_counter, english_counter = 0, 0  # Reset counters

        if any([x in line for x in skip_list]): 
            continue
        if contains_korean(line): continue
        if line:  # Check if line is not empty
            words = line.split()
            current_line = ''
            consecutive_breaks = 0  # Reset consecutive breaks count
            for word in words:
                if len(current_line) + len(word) + 1 <= max_line_length:
                    current_line += ' ' + word if current_line else word
                else:
                    wrapped_lines.append(current_line)
                    current_line = word
            if current_line:
                wrapped_lines.append(current_line)
        else:
            consecutive_breaks += 1
            if consecutive_breaks < 1:  # Allow up to two consecutive line breaks
                wrapped_lines.append('')
        #if english_counter == 1 and korean_counter == 1:
        #    if wrapped_lines[-1] is not '':
        #        wrapped_lines.append('')

    while wrapped_lines and wrapped_lines[0] == '':
        wrapped_lines.pop(0)

    return wrapped_lines


def make_pdf(self):
    font_path = 'fonts/arial-unicode-ms.ttf'
    image_folder = self.path['raw_combined']
    text_folder = self.path['text']
    pdf_output = os.path.join(self.final_direc, self.name+'.pdf')

    c = canvas.Canvas(pdf_output)
    
    sorted_filenames = sorted(os.listdir(text_folder), key=natural_sort_key)

    for filename in sorted_filenames:
        if filename.endswith('.txt'):
            image_path = os.path.join(image_folder, filename.replace('.txt', '.jpeg'))
            text_path = os.path.join(text_folder, filename)

            if not os.path.exists(image_path):
                continue

            with Image.open(image_path) as img:
                # Adjust font size based on image height
                font_size, line_height = scale_font_line_height(img.height)
                font = ImageFont.truetype(font_path, font_size)
                draw = ImageDraw.Draw(img)  # Create a draw object to measure text

                new_img = Image.new('RGB', (int(img.width * 2), int(1 * img.height)), "white")
                new_img.paste(img, (0, 0))
                draw = ImageDraw.Draw(new_img)

                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as file:
                        try:
                            text = ''.join(file.read().split('BEGIN_TEXT')[1])
                            text = text.split('END_TEXT')[0]
                        except:
                            print('failed to read response of : ', text_path)
                        wrapped_text = wrap_text(text, LINEWIDTH, font, draw)  # Set max_line_length as needed

                        y_offset = 15
                        for line in wrapped_text:
                            draw.text((img.width + 10, y_offset), line, fill="black", font=font)
                            y_offset += line_height # Adjust line spacing based on scaled line height


                temp_path = f'temp_{uuid.uuid4()}.jpg'
                new_img.save(temp_path)

                c.drawImage(temp_path, 0, 0, width=8.5*72, height=11*72, preserveAspectRatio=True, anchor='c')
                c.showPage()
                os.remove(temp_path)

    c.save()