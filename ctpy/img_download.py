from selenium import webdriver
from selenium.webdriver.common.by import By  # Import By for selecting elements
import time
import requests
import shutil
import os

def scrape(self):
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
    time.sleep(30)

    # Find all images on the page using the updated method
    images = driver.find_elements(By.TAG_NAME, 'img')

    def test_conditions(url):
        # test url string for violations
        # used to screen out images
        out = True
        if not any([ x in url for x in ['.jpeg','.jpg']]):
            out = False
        if not url.startswith(('http:', 'https:')):
            out = False                       
        return out

    img_urls = [x.get_attribute('src') for x in images]
    img_urls = [x for x in img_urls if test_conditions(x)]
    if 'manga-demon' in self.url:
        img_urls = [x for x in img_urls if 'Skeleton%20Soldier' in x]
    ind = 0
    # Download each image
    for img_url in img_urls:
        img_data = requests.get(img_url).content
        file_name = os.path.join(self.path['raw_image'], f'page_num_{ind}.jpeg')
        ind+=1            
        with open(file_name, 'wb') as f:
            f.write(img_data)
            print(f'Downloaded {img_url} to {file_name}')
        time.sleep(1)

    # Close the browser when done
    driver.quit()
    if not os.path.exists(self.path['raw_combined']):
        shutil.copytree(self.path['raw_image'],self.path['raw_combined'])