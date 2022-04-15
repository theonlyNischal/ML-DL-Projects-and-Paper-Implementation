"""Make sure to install the packages

# Install necessary packages
!pip install --upgrade --quiet Pillow

!pip install --quiet pdf2image
!sudo apt-get -qq install poppler-utils

!sudo apt -qq install tesseract-ocr-nep
!pip install --quiet pytesseract
"""

# Import necessary packages

import os
import numpy as np
import matplotlib.pyplot as plt
try:
  from PIL import Image
except ImportError:
  import Image

from google.colab import files
import pytesseract
from pdf2image import convert_from_path

def convert_pdf_to_imgs(pdf_file):
    """Converts a  pdf file to images
    """
    return convert_from_path(pdf_file)


def ocr_core(file, lang="nep"):
    """Image to string
    """
    custom_config = ' --oem 1 --psm 6'
    text = pytesseract.image_to_string(file, lang=lang, config=custom_config)
    return text

def get_searchable_pdf(pdf_file_path):
    # Todo: Check implementation 
    images = convert_pdf_to_imgs(pdf_file_path)
    with open('test.pdf', 'w+b') as f:
        for pg, img in enumerate(images):
            pdf = pytesseract.image_to_pdf_or_hocr(img, extension='pdf')
            f.write(pdf) # pdf type is bytes by default
    return pdf

def pdf_to_text(pdf_file_path, make_searchable=False):
    if make_searchable:
        #Not implemented
        pdf = get_searchable_pdf(pdf_file_path)
    images = convert_pdf_to_imgs(pdf_file_path)
    with open(f"{pdf_file_path}.txt", "w") as file:
        for pg, img in enumerate(images):
            print(ocr_core(img))
            file.write(ocr_core(img))

texts = pdf_to_text("path_to_pdf", make_searchable=False)