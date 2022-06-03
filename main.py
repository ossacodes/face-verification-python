import cv2
import face_recognition
import io
import urllib.request
import numpy as np
from typing import Union
from fastapi import FastAPI, File, UploadFile
app = FastAPI()


@app.get("/")
def read_root():
    req = urllib.request.urlopen(
        'https://assets.gqindia.com/photos/627cfe9f59a30320660c3e52/master/pass/Lionel%20Messi.jpeg')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'Load it as it is'

    # img = cv2.imread("Messi1.webp")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_encoding = face_recognition.face_encodings(rgb_img)[0]

    req2 = urllib.request.urlopen(
        'https://e0.365dm.com/21/05/2048x1152/skysports-lionel-messi-barcelona_5390329.jpg')
    arr2 = np.asarray(bytearray(req2.read()), dtype=np.uint8)
    img2 = cv2.imdecode(arr2, -1)  # 'Load it as it is'

    # img2 = cv2.imread("images/Jeff Bezoz.jpg")
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

    result = face_recognition.compare_faces([img_encoding], img_encoding2)
    print("Result: ", result)
    return {"Hello": "World", 'verification': str(result)}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

# cv2.imshow("Img", img)
# cv2.imshow("Img 2", img2)
# cv2.waitKey(0)
