import easyocr
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import os
import openai
import key
import webbrowser
from PIL import Image
from pytesseract import pytesseract

# ------------------------------------------------ #
# ---------EASYOCR Image Recognition-------------- #
# ------------------------------------------------ #

# True value set to GPU since easyocr performs quicker with a GPU
reader = easyocr.Reader(['en'], gpu=True)
# Control Test
#vid = cv2.VideoCapture("video.mp4")
# For Video Capture
vid = cv2.VideoCapture(0)
skip_frame = True

while True:
    _, image = vid.read()
    cv2.imshow('Text Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('test1.jpg', image)
        break
vid.release()
cv2.destroyAllWindows



path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
imagepath = 'test1.jpg'
pytesseract.tesseract_cmd=path_to_tesseract
text = pytesseract.image_to_string(Image.open(imagepath))
#print(text[:-1])
print(text)


# Read image segment
# Chosen font will be changed to something more appealing
# #while True:
# a = time.time()
# ret, img = vid.read()
#
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# result = reader.readtext(gray)
# text = ""
#
# for res in result:
#     text += res[1] + " "
#     cv2.putText(img, text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
#
# # FPS
# b = time.time()
# fps = 1 / (b - a)
# cv2.line(img, (20, 25), (127, 25), [85, 45, 255], 30)
# cv2.putText(img, text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
# cv2.imshow("result", img)
#
#     # if cv2.waitKey(1000) & 0XFF == ord('q'):
#     #     break
# # Output the FPS and recognized Text
# #     print(fps)
# #     print(text)
#
# # ------------------------------------------------ #
# # ---------DALL E 2 Image Generation-------------- #
# # ------------------------------------------------ #
# print(text)
# # ApiKey taken from OpenAI website, this key has been deleted
#
# Set apikey to following value
# List the model, we can choose which AI Model to use, 4 to choose from
# Create Image with parameters
# Reference Documentation at: https://beta.openai.com/docs/api-reference/images/create?lang=python
print(text)
openai.api_key = key.API_KEY
openai.Model.list()
response = openai.Image.create(
  prompt=text,
  n=1,
  size="1024x1024"
)

image_url = response['data'][0]['url']
webbrowser.open(image_url)
print('test')


