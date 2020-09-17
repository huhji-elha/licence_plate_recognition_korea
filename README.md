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

### More Issues
* Opencv(Yolo) + Pillow(한글 출력) 동시 사용 : Licence_Plate_detection.py
    * opencv로는 이미지 위에 한글 출력이 안됨. 그래서 PIL을 사용해야하는데 python 으로 opencv+pillow 결과 함께 출력하는 코드를 못찾아서 직접 짬.
    * 더 깔끔한 코드와 프로세스를 위해 연구중임.
* OCR API가 표지판 글자에는 저조한 인식률을 보임. 추가 학습 준비중. 
    * 현재 코드에서는 API가 글자 인식(약 60%) 정도의 정확도 --> 번호판 형태 정규표현식으로 필터링해서 화면에 띄우는 과정으로 진행됨.
    * 구글링해보니 아직 한국어 번호판 인식에 최적된 OCR 공개 모델은 없음. 
        1) 번호판이 개인정보로 분류되어 데이터 수집의 어려움. 
        2) 기본 OCR 모델이 잘 안먹히는 이유로 번호판에만 사용되는 특수한 폰트를 이유로 들 수 있음.
