import cv2
import os,io
import numpy as np
from functools import wraps
import time,pickle
import base64
from PIL import Image
import requests,ast,json
#from infer_east import east
from fastapi import FastAPI, HTTPException, Form,File,UploadFile
from collections import OrderedDict
from OCRDecode import ocr
app = FastAPI()
import random
@app.post("/ocr")

async def decodes(CameraID: str ,fileb: bytes = File(...),):

        final ={}
        final["received_time"] = time.strftime("%d-%m-%Y %H:%M:%S")
        
        
        img = np.array(Image.open(io.BytesIO(fileb)).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY ) 
      
        start = time.time()
        res = ocr.predict([img])[0]
        end = time.time()
        final["result"] = res
        final["inference_time"] = end - start
        return final,200







