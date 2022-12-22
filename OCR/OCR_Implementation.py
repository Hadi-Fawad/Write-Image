import easyocr
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

reader = easyocr.Reader(['en'], gpu=True)
# Control Test
vid = cv2.VideoCapture("video.mp4")
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

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    print(fps)
    print(text)



