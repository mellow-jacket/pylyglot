import os

class config:
    def __init__(self):
        self.abs_path = os.path.abspath(__file__).rstrip('//config.py')
        self.prompts = os.path.join(self.abs_path, 'prompt_templates')
        self.gpt_prompt = os.path.join(self.prompts,'gpt_base_prompt.txt')
        self.box_prompt = os.path.join(self.prompts,'redbox_base_prompt.txt')
        self.gpt_template = os.path.join(self.prompts,'response_template.txt')
        self.gpt_examples = os.path.join(self.prompts,'examples.txt')
        self.gpt_suffix = os.path.join(self.prompts,'gpt_base_suffix.txt')
        self.text_prompt = os.path.join(self.prompts,'gpt_text_prompt.txt')
        self.gemini_prompt = os.path.join(self.prompts,'gemini_base_prompt.txt')
        self.text_examples = os.path.join(self.prompts,'text_examples.txt')
        self.text_refine = os.path.join(self.prompts,'gpt_text_refine.txt')

        self.img_path = os.path.join(self.abs_path,'..')
        self.testimg_path = os.path.join(self.abs_path,'..','test_images')
        self.benchmark_path = os.path.join(self.abs_path,'..','benchmark')