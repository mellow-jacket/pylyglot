'''
OCR tools

mostly pytesseract

'''
import os
import json
from PIL import Image
import pytesseract
import easyocr
import numpy as np

def perform_ocr_on_all_images(self, lang='kor'):
    for image_name in os.listdir(self.path['image']):
        image_name = image_name.split('.')[0]
        #try:
        if True:
            image_path = os.path.join(self.path['ocr_image'], f'{image_name}.jpeg')
            boxes_path = os.path.join(self.path['box_coords'], f'{image_name}.txt')

            # Load boxes from saved file
            with open(boxes_path, 'r') as f:
                boxes = json.load(f)

            # Perform OCR on the boxes and save the results
            #image = Image.open(image_path)
            ocr_results = "OCR Analysis:\n"
            for box in boxes:
                box_text = ocr_on_box(image_path, box, lang=lang)
                ocr_results += box_text + '\n'
        #except:
        #    ocr_results = "OCR Analysis:\n"

        # Save OCR results
        with open(os.path.join(self.path['ocr_text'], f'{image_name}.txt'), 'w') as file:
            file.write(ocr_results)

def ocr_on_box(image_path, box, lang='kor'):
    """
    Perform OCR on a specific box of the image.
    """
    if lang == 'kor': 
        lang = 'ko'
    reader = easyocr.Reader([lang])
    # Crop the image to the box
    image = Image.open(image_path)
    cropped_image = np.array(image.crop((box[0][0], box[0][1], box[1][0], box[1][1])))
    
    # Perform OCR on the cropped image
    #text = pytesseract.image_to_string(cropped_image, lang=lang, config='--oem 3 --psm 11')
    #text = text.replace('\n', '').replace('PSM 11 Result:', '')
    # Use EasyOCR to read the text
    results = reader.readtext(cropped_image)
    # Process the results
    text = ' '.join([result[1] for result in results])
    text += '\n'
    return text

def old_ocr(self, lang = 'kor'):
    '''
    Pull possible korean text from processed image
    '''
    direc = 'image'
    paths = [ x for x in os.listdir(self.path[direc])]
    for name in paths:
        img_path = os.path.join(self.path[direc], name)
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img, lang=lang, config=f'--oem 2 --psm 4') #psm 3 or 11
        text = text.replace('\n','')
        text = text.replace('PSM 11 Result:','')
        text = 'OCR analysis :'+text+'\n'
        with open(os.path.join(self.path['ocr_text'], name.split('.')[0]+'.txt'), 'w') as file:
            file.write(text)

def ocr(self, lang = 'ko'):
    '''
    Pull possible Korean text from processed image using EasyOCR
    '''
    direc = 'image'
    paths = [x for x in os.listdir(self.path[direc])]
    
    # Create a reader instance for Korean language
    reader = easyocr.Reader([lang])

    for name in paths:
        img_path = os.path.join(self.path[direc], name)
        img = Image.open(img_path)
        
        # Use EasyOCR to read the text
        results = reader.readtext(img)
        
        # Process the results
        text = ' '.join([result[1] for result in results])
        text = f'Page {name.split(".")[0].split("_")[-1]}:{text}\n'
        
        # Write the results to a file
        with open(os.path.join(self.path['ocr_text'], name.split('.')[0] + '.txt'), 'w') as file:
            file.write(text)