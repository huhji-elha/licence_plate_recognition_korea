## Licence-Plate-Recognition

### Licence Plate Detection with YOLO

from Darknet, get pretrained YOLO v3 model and weights

    python object_detection_yolo.py --image test_image/car_snow.jpg

* detection된 번호판이 crop되도록 코드수정
* 동영상 input용 코드 추후 업데이트 예정

### Licence Plate Recognition with Google/Naver API

    python google_cloud_ocr.py

    python naver_cloud_ocr.py

* wild한 환경의 인식을 위해선 추가학습이 필요

* google과 naver 각 개인 보안 API KEY를 발급받아 사용 가능

----------------------------------------------------------------------------

### Licence Plate Detection with YOLO + Recognition with GoogleAPI
#### Image version


![image_output](Images/sample01.jpg)


#### Video version


![](Images/video_output_sample.gif)

-----------------------------------------------------------------------------

### Clear Issues
* Opencv(Yolo) + Pillow(한글 출력) 동시 사용 : Licence_Plate_detection.py
    * opencv로는 이미지 위에 한글 출력이 안됨. 그래서 PIL을 사용해야하는데 python 으로 opencv+pillow 결과롤 함께 출력하는 코드를 못찾아서 직접 짬.
    * 더 깔끔한 코드와 프로세스를 위해 연구중임.
* OCR API가 표지판 글자에는 저조한 인식률을 보임. 추가 학습 준비중.
