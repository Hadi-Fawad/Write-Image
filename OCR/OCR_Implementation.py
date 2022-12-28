import easyocr
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import os
import openai

# ------------------------------------------------ #
# ---------EASYOCR Image Recognition-------------- #
# ------------------------------------------------ #

# True value set to GPU since easyocr performs quicker with a GPU
reader = easyocr.Reader(['en'], gpu=True)
# Control Test
vid = cv2.VideoCapture("video.mp4")
# For Video Capture
# vid = cv2.VideoCapture(0)
skip_frame = True

# Read image segment
# Chosen font will be changed to something more appealing
# while True:
a = time.time()
ret, img = vid.read()

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
result = reader.readtext(gray)
text = ""

for res in result:
    text += res[1] + " "
cv2.putText(img, text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

# FPS
b = time.time()
fps = 1/(b-a)
cv2.line(img, (20, 25), (127, 25), [85, 45, 255], 30)
cv2.putText(img, text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
cv2.imshow("result", img)

    # if cv2.waitKey(1000) & 0XFF == ord('q'):
    #     break
# Output the FPS and recognized Text
#     print(fps)
#     print(text)

# ------------------------------------------------ #
# ---------DALL E 2 Image Generation-------------- #
# ------------------------------------------------ #
print(text)
# ApiKey taken from OpenAI website, this key has been deleted
apickey = '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'

# Set apikey to following value
# List the model, we can choose which AI Model to use, 4 to choose from
# Create Image with parameters
# Reference Documentation at: https://beta.openai.com/docs/api-reference/images/create?lang=python
openai.api_key = apickey
openai.Model.list()
object = openai.Image.create(
  prompt= text,
  n=1,
  size="1024x1024"
)

print(object)


