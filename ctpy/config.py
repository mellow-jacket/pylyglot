import os

class config:
    def __init__(self):
        self.abs_path = os.path.abspath(__file__).rstrip('//config.py')
        
        self.img_path = os.path.join(self.abs_path,'..')
        self.prompt_path = os.path.join(self.abs_path,'prompt.txt')
        self.template_path = os.path.join(self.abs_path,'response_template.txt')
        self.examples_path = os.path.join(self.abs_path,'examples.txt')
        self.suffix_path = os.path.join(self.abs_path,'prompt_suffix.txt')
        self.testimg_path = os.path.join(self.abs_path,'..','test_images')