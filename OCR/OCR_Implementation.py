import easyocr
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import os
import openai


reader = easyocr.Reader(['en'], gpu=True)
# Control Test
vid = cv2.VideoCapture("numplate.mp4")
# For Video Capture
# vid = cv2.VideoCapture(0)
skip_frame = True
 
while True:
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

    if cv2.waitKey(1000) & 0XFF == ord('q'):
        break
    print(fps)
    print(text)


apickey = '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'

openai.api_key = apickey
openai.Model.list()
openai.Image.create(
  prompt=text,
  n=2,
  size="1024x1024"

)


