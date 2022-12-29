import cv2
import openai
import const
import webbrowser
from PIL import Image
from pytesseract import pytesseract

# ------------------------------------------------ #
# ---------EASYOCR Image Recognition-------------- #
# ------------------------------------------------ #
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


imagepath = 'test1.jpg'
pytesseract.tesseract_cmd = const.PATH_TO_TESSERACT
text = pytesseract.image_to_string(Image.open(imagepath))

# # ------------------------------------------------ #
# # ---------DALL E 2 Image Generation-------------- #
# # ------------------------------------------------ #
# List the model, we can choose which AI Model to use, 4 to choose from
# Create Image with parameters
# Reference Documentation at: https://beta.openai.com/docs/api-reference/images/create?lang=python

print(text)
openai.api_key = const.API_KEY
openai.Model.list()
response = openai.Image.create(
  prompt=text,
  n=1,
  size="1024x1024"
)

image_url = response['data'][0]['url']
webbrowser.open(image_url)


