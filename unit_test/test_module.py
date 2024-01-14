import pytest
import pylyglot as ct
from dotenv import load_dotenv
import os

# Load environment variables for testing
load_dotenv()

# # Test if API_KEY is loaded correctly
# def test_api_key_loading():
#     api_key = os.getenv('API_KEY')
#     assert api_key is not None, "API_KEY should not be None"

URL = 'test_data\\comic'

# Test the initialization of the `issue` object
def test_issue_initialization():
    ex = ct.issue(
        name='unit_test',
        url=URL,  # None uses test_images
        api_key=os.getenv('API_KEY')
    )
    assert ex is not None, "Issue object should be initialized"

# Test the scrape method
def test_scrape():
    ex = ct.issue(
        name='unit_test',
        url=URL,
        api_key=os.getenv('API_KEY')
    )
    ex.scrape()
    assert True, "Scrape method has run successfully"

# Test downsampling
def test_downsample():
    ex = ct.issue(
        name='unit_test',
        url=URL,
        api_key=os.getenv('API_KEY')
    )
    ex.scrape()
    ex.downsample(scale_factor=1.0)
    assert True, "Downsample method has run successfully"

# Test the combine_gpt_and_ocr method
def test_combine_gpt_and_ocr():
    ex = ct.issue(
        name='unit_test',
        url=URL,
        api_key=os.getenv('API_KEY')
    )
    ex.scrape()
    ex.downsample(scale_factor=1.0)
    ex.combine_gpt_and_ocr()
    assert True, "combine_gpt_and_ocr method has run successfully"

# Test the translation method
def test_translate():
    ex = ct.issue(
        name='unit_test',
        url=URL,
        api_key=os.getenv('API_KEY')
    )
    ex.scrape()
    ex.downsample(scale_factor=1.0)
    ex.combine_gpt_and_ocr()
    ex.translate(debug=True)
    assert True, "Translate method has run successfully"

# Test PDF generation
def test_make_pdf():
    ex = ct.issue(
        name='unit_test',
        url=URL,
        api_key=os.getenv('API_KEY')
    )
    ex.scrape()
    ex.downsample(scale_factor=1.0)
    ex.combine_gpt_and_ocr()
    ex.translate(debug=True)
    ex.make_pdf()
    assert True, "PDF generation method has run successfully"
