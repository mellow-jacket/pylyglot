from selenium import webdriver
from selenium.webdriver.common.by import By  # Import By for selecting elements
import time
import requests
import shutil
import os
import re
from PIL import Image
from io import BytesIO


def scrape(self):
    # Set up the Selenium WebDriver. The example uses Chrome.
    # Set up the Selenium WebDriver. The example uses Chrome.
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    #options.add_argument('--headless')  # Optionally, run Chrome in headless mode (without GUI)

    driver = webdriver.Chrome(options=options)

    # URL of the page you want to scrape
    # 260 url
    #url = 'https://newtoki317.com/webtoon/33111401?toon=%EC%9D%BC%EB%B0%98%EC%9B%B9%ED%88%B0'

    # Navigate to the page
    driver.get(self.url)
    # Give the page some time to load
    time.sleep(20)

    # Find all images on the page using the updated method
    images = driver.find_elements(By.TAG_NAME, 'img')

    def test_condition(text):
        text = str(text)
        return bool(re.search(r'novel.*\.(jpeg|jpg|png)$', text))

    img_urls = [x.get_attribute('src') for x in images]
    img_urls = [ x for x in img_urls if test_condition(x)]

    ind = 0
    # Download each image and convert to jpeg
    for img_url in img_urls:
        # Fetch the image
        response = requests.get(img_url)
        img_data = response.content

        # Convert the image to jpeg using Pillow
        try:
            with Image.open(BytesIO(img_data)) as img:
                file_name = os.path.join(self.path['scraped_image'], f'page_num_{ind}.jpeg')
                img.convert('RGB').save(file_name, 'JPEG')
                print(f'Converted and saved {img_url} to {file_name}')
        except IOError:
            print(f"Cannot convert image: {img_url}")
        
        ind += 1
        time.sleep(1)

    # Close the browser when done
    driver.quit()
    if not os.path.exists(self.path['raw_combined']):
        shutil.copytree(self.path['scraped_image'],self.path['raw_combined'])