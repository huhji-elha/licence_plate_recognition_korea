## opencv + PIL 연습 코드

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import os, io

from PIL import ImageFont, ImageDraw, Image

parser = argparse.ArgumentParser(
    description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

def detect_text(path):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    #print(type(content))

    
    image = vision.types.Image(content=content)
    print(image)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts, content

api_output, _ = detect_text(args.image)

recoq = []
for text in api_output :
    recoq.append(text.description)

img = cv.imread(args.image, cv.IMREAD_COLOR)
#print(type(img))
b,g,r,a = 0,0,255,0
fontpath = "/font path to/NanumGothic.ttf"
font = ImageFont.truetype(fontpath, 80)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((10, 10),  recoq[0], font=font, fill=(b,g,r,a))

img = np.array(img_pil)
#cv.putText(img,  "", (10,10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv.LINE_AA)

cv.imwrite("path to output folder/output.jpg", img)
