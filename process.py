
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

# # Process inputs
# winName = 'Deep learning object detection in OpenCV'
# cv.namedWindow(winName, cv.WINDOW_NORMAL)

# outputFile = "yolo_out_py.avi"
# if (args.image):
#     # Open the image file
#     if not os.path.isfile(args.image):
#         print("Input image file ", args.image, " doesn't exist")
#         sys.exit(1)
#     cap = cv.VideoCapture(args.image)
#     outputFile = args.image[:-4]+'_yolo_out_py.jpg'
# elif (args.video):
#     # Open the video file
#     if not os.path.isfile(args.video):
#         print("Input video file ", args.video, " doesn't exist")
#         sys.exit(1)
#     cap = cv.VideoCapture(args.video)
#     outputFile = args.video[:-4]+'_yolo_out_py2.avi'
# else:
#     # Webcam input
#     cap = cv.VideoCapture(0)


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

#image = "/home/huhji/licence_plate/test_image/car_big_LP_04.jpeg"
api_output, _ = detect_text(args.image)

recoq = []
for text in api_output :
    recoq.append(text.description)

print(recoq[0])

img = cv.imread(args.image, cv.IMREAD_COLOR)
#print(type(img))
b,g,r,a = 0,0,255,0
fontpath = "/home/huhji/.local/share/fonts/NanumGothic.ttf"
font = ImageFont.truetype(fontpath, 80)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((10, 10),  recoq[0], font=font, fill=(b,g,r,a))

img = np.array(img_pil)
#cv.putText(img,  "", (10,10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r), 1, cv.LINE_AA)

cv.imwrite("/home/huhji/licence_plate/output_sample.jpg", img)