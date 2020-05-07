import json
import base64
import requests

with open("test_image/car_snow.jpg", 'rb') as f :
    img = base64.b64encode(f.read())

URL = "APIGW Invoke URL"

KEY = "Naver OCR Secret Key"

headers = {
    "Content-Type" : "application/json",
    "X-OCR-SECRET" : KEY
}

data = {
    "version" : "V1",
    "requestID" : "sample_id",
    "timestamp" : 0,
    "images" : [
        {
            "name" : "sample_image",
            "format" : "jpg",
            "data" : img.decode('utf-8')
        }
    ]
}

data = json.dumps(data)
response = requests.post(URL, data=data, headers=headers)
res = json.loads(response.text)
