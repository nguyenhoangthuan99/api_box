from datetime import datetime, timedelta
from typing import Optional
import pickle, json
import base64,cv2
import jetson.utils

with open('config.json', 'r') as fp:
    config = json.load(fp)

fake_users_db = config["users"]
list_camera = config["cameras"]

camera_active = {}

for key in list_camera.keys():
    if list_camera[key]["status"] == "1":
        print("CAPTURING ",list_camera[key]["rstp_link"])
        gst_str = ('rtspsrc location={} latency=0 ! queue max-size-buffers=1 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True').format(list_camera[key]["rstp_link"])
        camera_active[key] = cv2.VideoCapture(gst_str)



fp = open('config.json', 'r')
version = json.load(fp)["version"]
fp.close()

import os,io
import numpy as np
import time,pickle
import base64
from PIL import Image
import requests,ast,json
from fastapi import FastAPI, HTTPException, Form,File,UploadFile, WebSocket,Depends, status
from collections import OrderedDict
from datetime import datetime
from end2end import inference
from starlette.websockets import WebSocket, WebSocketDisconnect

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from item import *

import eventlet
eventlet.monkey_patch()

def d(x1,y1,x2,y2):
   return ((x1-x2)**2  +(y1-y2)**2)**0.5
app = FastAPI()
import random

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Notifier:
    def __init__(self):
        self.connections: List[WebSocket] = []
        self.generator = self.get_notification_generator()

    async def get_notification_generator(self):
        while True:
            message = yield
            await self._notify(message)

    async def push(self, msg: dict):
        await self.generator.asend(str(msg))

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
    
    async def remove(self, websocket: WebSocket):
        self.connections.remove(websocket)

    async def _notify(self, message: str):
        living_connections = []
        while len(self.connections) > 0:
            # Looping like this is necessary in case a disconnection is handled
            # during await websocket.send_text(message)
            websocket = self.connections.pop()
            await websocket.send_text(message)
            living_connections.append(websocket)
        self.connections = living_connections

notifier = Notifier()
@app.post("/stream_decode_gstreamer")
async def stream_decodes_gstreamer(CameraID: str ):
        global capture
        final ={}
        final["received_time"] = time.strftime("%d-%m-%Y %H:%M:%S")
        if CameraID not in camera_active.keys() and list_camera[CameraID]["status"] == "1":
            return {"error": "CameraID is not active"} ,0      
        
       # camera_active[CameraID].set(cv2.CAP_PROP_BUFFERSIZE, 1);
       # camera_active[CameraID].set(cv2.CAP_PROP_POS_FRAMES, -2)
        ret, img = camera_active[CameraID].read()
        
        # Try to read last frame again 6 times
        count = 0
        
        while (1):
            if (not ret) and (count < 1):
               # camera_active[CameraID].set(cv2.CAP_PROP_POS_FRAMES, -2)
                ret, img = camera_active[CameraID].read()
                count += 1
                print("------ Tried to fix ", count," times")
            else:
                break

        # if can not read again ==> re-connect to camera
        if not ret:
            gst_str = ('rtspsrc location={} latency=0 ! queue max-size-buffers=1 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True').format(list_camera[CameraID]["rstp_link"])
            camera_active[CameraID] = cv2.VideoCapture(gst_str)
            
            
            print("------------- RECONNNECT TO CAMERA ---------------")
            ret, img = camera_active[CameraID].read()
            if not ret:
                return {"error": "cannot connect to video's stream"}, 0
        
        img = img[:,:,::-1]
        img_ori = img.copy()
        start = time.time()
        ################ run infer image ##############

        ocr_res,bbox,img = inference.run(img)

        ###############################################
        ## drawing result
        for i in range(np.array(bbox).shape[0]):
          #  print(bbox[i].shape)
            tmp = bbox[i][2].copy()
            bbox[i][2] = bbox[i][3]
            bbox[i][3] = tmp
            cv2.polylines(img, [bbox[i].astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
            img[:,:,::-1] = cv2.putText(img,ocr_res[i], (bbox[i][1][0],bbox[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0,0,255), 2, cv2.LINE_AA) 
        if False:
            cv2.imwrite("test.jpg",img[:,:,::-1])
        img = cv2.resize(img,(640,480))
        res_name_ = str(time.time())+".jpg"
        #cv2.imwrite("logs/"+res_name_,img)
        ### buil final result
        # OrderedDict()
        final["CameraID"] = CameraID
        final["AiboxID"] = "PROTOTYPE-01"
        tmp=Image.fromarray(np.uint8(cv2.resize(img_ori,(640,480))))
        b = io.BytesIO()
        tmp.save(b, 'jpeg')
        tmp = b.getvalue()
        final["input_image"] = base64.b64encode(tmp)
        

        tmp=Image.fromarray(np.uint8(img))
        b = io.BytesIO()
        tmp.save(b, 'jpeg')
        tmp = b.getvalue()
        final["output_image"] =base64.b64encode(tmp)


        for i in range(np.array(bbox).shape[0]):
             ids = str(i+1)
             bb_name = "text_box"+ids
             lbl_name = "text_label"+ids
             final[bb_name]=str([bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]])
             final[lbl_name] = ocr_res[i]
        #final["detect_time"] = str(end-start)
        #final["ocr_time"] = str(end2-end)
        
        plate_res = []
        if len(bbox) == 1:
            final["plate_box1"] = str([bbox[0][0][0],bbox[0][0][1],bbox[0][1][0],bbox[0][1][1],bbox[0][2][0],bbox[0][2][1],bbox[0][3][0] ,bbox[0][3][1]])
            final["plate_text1"] = ocr_res[0]
        elif len(bbox) > 1:
   
          for i in range(np.array(bbox).shape[0]):
            for j in range(np.array(bbox).shape[0]):
                if i != j :
                    x1,y1,x2,y2,x3,y3,x4,y4 = bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]
                    x10,y10,x20,y20,x30,y30,x40,y40 = bbox[j][0][0],bbox[j][0][1],bbox[j][1][0],bbox[j][1][1],bbox[j][2][0],bbox[j][2][1],bbox[j][3][0] ,bbox[j][3][1]
                    X0 = (x1+x2+x3+x4)/4.
                    Y0 = (y1+y2+y3+y4)/4.
                    X1 = (x10+x20+x30+x40)/4.
                    Y1 = (y10+y20+y30+y40)/4.
                    if d(X0,Y0,X1,Y1) < 1.5*min(d(x1,y1,x4,y4),d(x10,y10,y30,y40)) and ((i,j) not in plate_res) and ((j,i) not in plate_res):
                        plate_res.append((i,j))
        done = []
        idx = 0
        for i in range(len(plate_res)):
            idx = i +1
            k,v = plate_res[i]
            done.append(k)
            done.append(v)
            key_name = "plate_box" +str(idx)
            key_txt = "plate_text"+str(idx)
            box1 = bbox[k]
            box2 = bbox[v]
            if box1[0][1] < box2[0][1]:
                less = box1
                greater = box2
                res_xxx =  ocr_res[k] +" "+  ocr_res[v]
            else: 
                less = box2
                greater = box1
                res_xxx =  ocr_res[v] +" "+  ocr_res[k]
            final[key_name] = str([less[0][0],less[0][1],less[1][0],less[1][1],greater[2][0],greater[2][1],greater[3][0] ,greater[3][1]])
            final[key_txt] = res_xxx
        for i in range(np.array(bbox).shape[0]):
            if i not in done:
                idx+=1
                key_name = "plate_box" +str(idx)
                key_txt = "plate_text"+str(idx)
                final[key_name] =str([bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]])
                final[key_txt] = ocr_res[i]
       # with open("logs_txt/" +time.strftime("%d-%m-%Y")+"/"+ res_name_ + "logs.pkl","wb") as logger:
        #    pickle.dump(final,logger, pickle.HIGHEST_PROTOCOL)
        final["latency"] = time.time() - start

        await notifier.push(final)


        return final, 200
@app.post("/stream_decode")
async def stream_decodes(CameraID: str ):
        global capture
        final ={}
        final["received_time"] = time.strftime("%d-%m-%Y %H:%M:%S")
        if CameraID not in camera_active.keys():
            return {"error": "CameraID is not active"} ,0      
        
        # camera_active[CameraID].set(cv2.CAP_PROP_BUFFERSIZE, 1);
        camera_active[CameraID].set(cv2.CAP_PROP_POS_FRAMES, -2)
        ret, img = camera_active[CameraID].read()
        
        # Try to read last frame again 6 times
        count = 0
        while (1):
            if (not ret) and (count < 6):
                camera_active[CameraID].set(cv2.CAP_PROP_POS_FRAMES, -2)
                ret, img = camera_active[CameraID].read()
                count += 1
                print("------ Tried to fix ", count," times")
            else:
                break

        # if can not read again ==> re-connect to camera
        if not ret:
            camera_active[CameraID] = cv2.VideoCapture(list_camera[CameraID]["rstp_link"])
            # camera_active[CameraID].set(cv2.CAP_PROP_BUFFERSIZE, 1);
            camera_active[CameraID].set(cv2.CAP_PROP_POS_FRAMES, -2)
            print("------------- RECONNNECT TO CAMERA ---------------")
            ret, img = camera_active[CameraID].read()
            if not ret:
                return {"error": "cannot connect to video's stream"}, 0

        img = img[:,:,::-1]
        img_ori = img.copy()
        start = time.time()
        ################ run infer image ##############

        ocr_res,bbox,img = inference.run(img)

        ###############################################
        ## drawing result
        for i in range(np.array(bbox).shape[0]):
          #  print(bbox[i].shape)
            tmp = bbox[i][2].copy()
            bbox[i][2] = bbox[i][3]
            bbox[i][3] = tmp
            cv2.polylines(img, [bbox[i].astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
            img[:,:,::-1] = cv2.putText(img,ocr_res[i], (bbox[i][1][0],bbox[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0,0,255), 2, cv2.LINE_AA) 
        if False:
            cv2.imwrite("test.jpg",img[:,:,::-1])
        img = cv2.resize(img,(640,480))
        res_name_ = str(time.time())+".jpg"
        #cv2.imwrite("logs/"+res_name_,img)
        ### buil final result
        # OrderedDict()
        final["CameraID"] = CameraID
        final["AiboxID"] = "PROTOTYPE-01"
        tmp=Image.fromarray(np.uint8(cv2.resize(img_ori,(640,480))))
        b = io.BytesIO()
        tmp.save(b, 'jpeg')
        tmp = b.getvalue()
        final["input_image"] = base64.b64encode(tmp)
        

        tmp=Image.fromarray(np.uint8(img))
        b = io.BytesIO()
        tmp.save(b, 'jpeg')
        tmp = b.getvalue()
        final["output_image"] =base64.b64encode(tmp)


        for i in range(np.array(bbox).shape[0]):
             ids = str(i+1)
             bb_name = "text_box"+ids
             lbl_name = "text_label"+ids
             final[bb_name]=str([bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]])
             final[lbl_name] = ocr_res[i]
        #final["detect_time"] = str(end-start)
        #final["ocr_time"] = str(end2-end)
        
        plate_res = []
        if len(bbox) == 1:
            final["plate_box1"] = str([bbox[0][0][0],bbox[0][0][1],bbox[0][1][0],bbox[0][1][1],bbox[0][2][0],bbox[0][2][1],bbox[0][3][0] ,bbox[0][3][1]])
            final["plate_text1"] = ocr_res[0]
        elif len(bbox) > 1:
   
          for i in range(np.array(bbox).shape[0]):
            for j in range(np.array(bbox).shape[0]):
                if i != j :
                    x1,y1,x2,y2,x3,y3,x4,y4 = bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]
                    x10,y10,x20,y20,x30,y30,x40,y40 = bbox[j][0][0],bbox[j][0][1],bbox[j][1][0],bbox[j][1][1],bbox[j][2][0],bbox[j][2][1],bbox[j][3][0] ,bbox[j][3][1]
                    X0 = (x1+x2+x3+x4)/4.
                    Y0 = (y1+y2+y3+y4)/4.
                    X1 = (x10+x20+x30+x40)/4.
                    Y1 = (y10+y20+y30+y40)/4.
                    if d(X0,Y0,X1,Y1) < 1.5*min(d(x1,y1,x4,y4),d(x10,y10,y30,y40)) and ((i,j) not in plate_res) and ((j,i) not in plate_res):
                        plate_res.append((i,j))
        done = []
        idx = 0
        for i in range(len(plate_res)):
            idx = i +1
            k,v = plate_res[i]
            done.append(k)
            done.append(v)
            key_name = "plate_box" +str(idx)
            key_txt = "plate_text"+str(idx)
            box1 = bbox[k]
            box2 = bbox[v]
            if box1[0][1] < box2[0][1]:
                less = box1
                greater = box2
                res_xxx =  ocr_res[k] +" "+  ocr_res[v]
            else: 
                less = box2
                greater = box1
                res_xxx =  ocr_res[v] +" "+  ocr_res[k]
            final[key_name] = str([less[0][0],less[0][1],less[1][0],less[1][1],greater[2][0],greater[2][1],greater[3][0] ,greater[3][1]])
            final[key_txt] = res_xxx
        for i in range(np.array(bbox).shape[0]):
            if i not in done:
                idx+=1
                key_name = "plate_box" +str(idx)
                key_txt = "plate_text"+str(idx)
                final[key_name] =str([bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]])
                final[key_txt] = ocr_res[i]
       # with open("logs_txt/" +time.strftime("%d-%m-%Y")+"/"+ res_name_ + "logs.pkl","wb") as logger:
        #    pickle.dump(final,logger, pickle.HIGHEST_PROTOCOL)
        final["latency"] = time.time() - start

        await notifier.push(final)


        return final, 200

@app.post("/decode2")
async def decodes2(body:Decode2):

        final ={}
        final["received_time"] = time.strftime("%d-%m-%Y %H:%M:%S")

        fileb = base64.b64decode(body.image)
        img = np.array(Image.open(io.BytesIO(fileb)).convert("RGB"))
        img_ori = img.copy()
        start = time.time()
        ################ run infer image ##############

        ocr_res,bbox,img = inference.run(img)

        ###############################################
        ## drawing result
        for i in range(np.array(bbox).shape[0]):
          #  print(bbox[i].shape)
            tmp = bbox[i][2].copy()
            bbox[i][2] = bbox[i][3]
            bbox[i][3] = tmp
            cv2.polylines(img, [bbox[i].astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
            img = cv2.putText(img,ocr_res[i], (bbox[i][1][0],bbox[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0,0,255), 2, cv2.LINE_AA) 
        if False:
            cv2.imwrite("test.jpg",img[:,:,::-1])
        img = cv2.resize(img,(640,480))
        res_name_ = str(time.time())+".jpg"
        #cv2.imwrite("logs/"+res_name_,img)
        ### buil final result
        # OrderedDict()
        final["CameraID"] = body.CameraID
        final["AiboxID"] = "PROTOTYPE-01"
        tmp=Image.fromarray(np.uint8(cv2.resize(img_ori,(640,480))))
        b = io.BytesIO()
        tmp.save(b, 'jpeg')
        tmp = b.getvalue()
        final["input_image"] = base64.b64encode(tmp)
        

        tmp=Image.fromarray(np.uint8(img))
        b = io.BytesIO()
        tmp.save(b, 'jpeg')
        tmp = b.getvalue()
        final["output_image"] =base64.b64encode(tmp)


        for i in range(np.array(bbox).shape[0]):
             ids = str(i+1)
             bb_name = "text_box"+ids
             lbl_name = "text_label"+ids
             final[bb_name]=str([bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]])
             final[lbl_name] = ocr_res[i]
        #final["detect_time"] = str(end-start)
        #final["ocr_time"] = str(end2-end)
        
        plate_res = []
        if len(bbox) == 1:
            final["plate_box1"] = str([bbox[0][0][0],bbox[0][0][1],bbox[0][1][0],bbox[0][1][1],bbox[0][2][0],bbox[0][2][1],bbox[0][3][0] ,bbox[0][3][1]])
            final["plate_text1"] = ocr_res[0]
        elif len(bbox) > 1:
   
          for i in range(np.array(bbox).shape[0]):
            for j in range(np.array(bbox).shape[0]):
                if i != j :
                    x1,y1,x2,y2,x3,y3,x4,y4 = bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]
                    x10,y10,x20,y20,x30,y30,x40,y40 = bbox[j][0][0],bbox[j][0][1],bbox[j][1][0],bbox[j][1][1],bbox[j][2][0],bbox[j][2][1],bbox[j][3][0] ,bbox[j][3][1]
                    X0 = (x1+x2+x3+x4)/4.
                    Y0 = (y1+y2+y3+y4)/4.
                    X1 = (x10+x20+x30+x40)/4.
                    Y1 = (y10+y20+y30+y40)/4.
                    if d(X0,Y0,X1,Y1) < 1.5*min(d(x1,y1,x4,y4),d(x10,y10,y30,y40)) and ((i,j) not in plate_res) and ((j,i) not in plate_res):
                        plate_res.append((i,j))
        done = []
        idx = 0
        for i in range(len(plate_res)):
            idx = i +1
            k,v = plate_res[i]
            done.append(k)
            done.append(v)
            key_name = "plate_box" +str(idx)
            key_txt = "plate_text"+str(idx)
            box1 = bbox[k]
            box2 = bbox[v]
            if box1[0][1] < box2[0][1]:
                less = box1
                greater = box2
                res_xxx =  ocr_res[k] +" "+  ocr_res[v]
            else: 
                less = box2
                greater = box1
                res_xxx =  ocr_res[v] +" "+  ocr_res[k]
            final[key_name] = str([less[0][0],less[0][1],less[1][0],less[1][1],greater[2][0],greater[2][1],greater[3][0] ,greater[3][1]])
            final[key_txt] = res_xxx
        for i in range(np.array(bbox).shape[0]):
            if i not in done:
                idx+=1
                key_name = "plate_box" +str(idx)
                key_txt = "plate_text"+str(idx)
                final[key_name] =str([bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]])
                final[key_txt] = ocr_res[i]
       # with open("logs_txt/" +time.strftime("%d-%m-%Y")+"/"+ res_name_ + "logs.pkl","wb") as logger:
        #    pickle.dump(final,logger, pickle.HIGHEST_PROTOCOL)
        final["latency"] = time.time() - start

        await notifier.push(final)

        return final, 200

        return final, 200


@app.post("/decode")
async def decodes(CameraID: str , fileb: bytes = File(...),):
        final ={}
        final["received_time"] = time.strftime("%d-%m-%Y %H:%M:%S")
       # if not os.path.exists("logs_txt/"+time.strftime("%d-%m-%Y")):
       #     os.mkdir("logs_txt/"+time.strftime("%d-%m-%Y"))
        
        img = np.array(Image.open(io.BytesIO(fileb)).convert("RGB"))
        img_ori = img.copy()
        start = time.time()
        ################ run infer image ##############

        ocr_res,bbox,img = inference.run(img)

        ###############################################
        ## drawing result
        for i in range(np.array(bbox).shape[0]):
          #  print(bbox[i].shape)
            tmp = bbox[i][2].copy()
            bbox[i][2] = bbox[i][3]
            bbox[i][3] = tmp
            cv2.polylines(img, [bbox[i].astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
            img = cv2.putText(img,ocr_res[i], (bbox[i][1][0],bbox[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0,0,255), 2, cv2.LINE_AA) 
        if False:
            cv2.imwrite("test.jpg",img[:,:,::-1])
        img = cv2.resize(img,(640,480))
        res_name_ = str(time.time())+".jpg"
        #cv2.imwrite("logs/"+res_name_,img)
        ### buil final result
        # OrderedDict()
        final["CameraID"] = CameraID
        final["AiboxID"] = "PROTOTYPE-01"
        tmp=Image.fromarray(np.uint8(cv2.resize(img_ori,(640,480))))
        b = io.BytesIO()
        tmp.save(b, 'jpeg')
        tmp = b.getvalue()
        final["input_image"] = base64.b64encode(tmp)
        

        tmp=Image.fromarray(np.uint8(img))
        b = io.BytesIO()
        tmp.save(b, 'jpeg')
        tmp = b.getvalue()
        final["output_image"] =base64.b64encode(tmp)


        for i in range(np.array(bbox).shape[0]):
             ids = str(i+1)
             bb_name = "text_box"+ids
             lbl_name = "text_label"+ids
             final[bb_name]=str([bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]])
             final[lbl_name] = ocr_res[i]
        #final["detect_time"] = str(end-start)
        #final["ocr_time"] = str(end2-end)
        
        plate_res = []
        if len(bbox) == 1:
            final["plate_box1"] = str([bbox[0][0][0],bbox[0][0][1],bbox[0][1][0],bbox[0][1][1],bbox[0][2][0],bbox[0][2][1],bbox[0][3][0] ,bbox[0][3][1]])
            final["plate_text1"] = ocr_res[0]
        elif len(bbox) > 1:
   
          for i in range(np.array(bbox).shape[0]):
            for j in range(np.array(bbox).shape[0]):
                if i != j :
                    x1,y1,x2,y2,x3,y3,x4,y4 = bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]
                    x10,y10,x20,y20,x30,y30,x40,y40 = bbox[j][0][0],bbox[j][0][1],bbox[j][1][0],bbox[j][1][1],bbox[j][2][0],bbox[j][2][1],bbox[j][3][0] ,bbox[j][3][1]
                    X0 = (x1+x2+x3+x4)/4.
                    Y0 = (y1+y2+y3+y4)/4.
                    X1 = (x10+x20+x30+x40)/4.
                    Y1 = (y10+y20+y30+y40)/4.
                    if d(X0,Y0,X1,Y1) < 1.5*min(d(x1,y1,x4,y4),d(x10,y10,y30,y40)) and ((i,j) not in plate_res) and ((j,i) not in plate_res):
                        plate_res.append((i,j))
        done = []
        idx = 0
        for i in range(len(plate_res)):
            idx = i +1
            k,v = plate_res[i]
            done.append(k)
            done.append(v)
            key_name = "plate_box" +str(idx)
            key_txt = "plate_text"+str(idx)
            box1 = bbox[k]
            box2 = bbox[v]
            if box1[0][1] < box2[0][1]:
                less = box1
                greater = box2
                res_xxx =  ocr_res[k] +" "+  ocr_res[v]
            else: 
                less = box2
                greater = box1
                res_xxx =  ocr_res[v] +" "+  ocr_res[k]
            final[key_name] = str([less[0][0],less[0][1],less[1][0],less[1][1],greater[2][0],greater[2][1],greater[3][0] ,greater[3][1]])
            final[key_txt] = res_xxx
        for i in range(np.array(bbox).shape[0]):
            if i not in done:
                idx+=1
                key_name = "plate_box" +str(idx)
                key_txt = "plate_text"+str(idx)
                final[key_name] =str([bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]])
                final[key_txt] = ocr_res[i]
       # with open("logs_txt/" +time.strftime("%d-%m-%Y")+"/"+ res_name_ + "logs.pkl","wb") as logger:
        #    pickle.dump(final,logger, pickle.HIGHEST_PROTOCOL)
        final["latency"] = time.time() - start

        await notifier.push(final)


        return final, 200

@app.post("/decode2")
async def decodes2(body:Decode2):

        final ={}
        final["received_time"] = time.strftime("%d-%m-%Y %H:%M:%S")

        fileb = base64.b64decode(body.image)
        img = np.array(Image.open(io.BytesIO(fileb)).convert("RGB"))
        img_ori = img.copy()
        start = time.time()
        ################ run infer image ##############

        ocr_res,bbox,img = inference.run(img)

        ###############################################
        ## drawing result
        for i in range(np.array(bbox).shape[0]):
          #  print(bbox[i].shape)
            tmp = bbox[i][2].copy()
            bbox[i][2] = bbox[i][3]
            bbox[i][3] = tmp
            cv2.polylines(img, [bbox[i].astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
            img = cv2.putText(img,ocr_res[i], (bbox[i][1][0],bbox[i][1][1]), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0,0,255), 2, cv2.LINE_AA) 
        if False:
            cv2.imwrite("test.jpg",img[:,:,::-1])
        img = cv2.resize(img,(640,480))
        res_name_ = str(time.time())+".jpg"
        #cv2.imwrite("logs/"+res_name_,img)
        ### buil final result
        # OrderedDict()
        final["CameraID"] = body.CameraID
        final["AiboxID"] = "PROTOTYPE-01"
        tmp=Image.fromarray(np.uint8(img_ori))
        b = io.BytesIO()
        tmp.save(b, 'jpeg')
        tmp = b.getvalue()
        final["input_image"] = base64.b64encode(tmp)
        

        tmp=Image.fromarray(np.uint8(img))
        b = io.BytesIO()
        tmp.save(b, 'jpeg')
        tmp = b.getvalue()
        final["output_image"] =base64.b64encode(tmp)


        for i in range(np.array(bbox).shape[0]):
             ids = str(i+1)
             bb_name = "text_box"+ids
             lbl_name = "text_label"+ids
             final[bb_name]=str([bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]])
             final[lbl_name] = ocr_res[i]
        #final["detect_time"] = str(end-start)
        #final["ocr_time"] = str(end2-end)
        
        plate_res = []
        if len(bbox) == 1:
            final["plate_box1"] = str([bbox[0][0][0],bbox[0][0][1],bbox[0][1][0],bbox[0][1][1],bbox[0][2][0],bbox[0][2][1],bbox[0][3][0] ,bbox[0][3][1]])
            final["plate_text1"] = ocr_res[0]
        elif len(bbox) > 1:
   
          for i in range(np.array(bbox).shape[0]):
            for j in range(np.array(bbox).shape[0]):
                if i != j :
                    x1,y1,x2,y2,x3,y3,x4,y4 = bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]
                    x10,y10,x20,y20,x30,y30,x40,y40 = bbox[j][0][0],bbox[j][0][1],bbox[j][1][0],bbox[j][1][1],bbox[j][2][0],bbox[j][2][1],bbox[j][3][0] ,bbox[j][3][1]
                    X0 = (x1+x2+x3+x4)/4.
                    Y0 = (y1+y2+y3+y4)/4.
                    X1 = (x10+x20+x30+x40)/4.
                    Y1 = (y10+y20+y30+y40)/4.
                    if d(X0,Y0,X1,Y1) < 1.5*min(d(x1,y1,x4,y4),d(x10,y10,y30,y40)) and ((i,j) not in plate_res) and ((j,i) not in plate_res):
                        plate_res.append((i,j))
        done = []
        idx = 0
        for i in range(len(plate_res)):
            idx = i +1
            k,v = plate_res[i]
            done.append(k)
            done.append(v)
            key_name = "plate_box" +str(idx)
            key_txt = "plate_text"+str(idx)
            box1 = bbox[k]
            box2 = bbox[v]
            if box1[0][1] < box2[0][1]:
                less = box1
                greater = box2
                res_xxx =  ocr_res[k] +" "+  ocr_res[v]
            else: 
                less = box2
                greater = box1
                res_xxx =  ocr_res[v] +" "+  ocr_res[k]
            final[key_name] = str([less[0][0],less[0][1],less[1][0],less[1][1],greater[2][0],greater[2][1],greater[3][0] ,greater[3][1]])
            final[key_txt] = res_xxx
        for i in range(np.array(bbox).shape[0]):
            if i not in done:
                idx+=1
                key_name = "plate_box" +str(idx)
                key_txt = "plate_text"+str(idx)
                final[key_name] =str([bbox[i][0][0],bbox[i][0][1],bbox[i][1][0],bbox[i][1][1],bbox[i][2][0],bbox[i][2][1],bbox[i][3][0] ,bbox[i][3][1]])
                final[key_txt] = ocr_res[i]
       # with open("logs_txt/" +time.strftime("%d-%m-%Y")+"/"+ res_name_ + "logs.pkl","wb") as logger:
        #    pickle.dump(final,logger, pickle.HIGHEST_PROTOCOL)
        final["latency"] = time.time() - start

        await notifier.push(final)

        return final, 200


@app.websocket("/plate-recognition")
async def websocket_endpoint(websocket: WebSocket):
    await notifier.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
           # await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        await notifier.remove(websocket)
   

@app.on_event("startup")
async def startup():
    # Prime the push notification generator
    await notifier.generator.asend(None)

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

async def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

async def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

async def authenticate_user(fake_db, username: str, password: str):
    user =  await get_user(fake_db, username)
    if not user:
        return False
    if not await verify_password(password, user.hashed_password):
        return False
    return user

async def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = await get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = await create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/getVersion")
async def getVersion():
    fp = open('config.json', 'r')
    version = json.load(fp)["version"]
    fp.close()
    return version

@app.post("/changePassword")
async def changePassword(body: ChangePassword ,current_user: User = Depends(get_current_active_user)):
    if current_user.username != body.username or not await verify_password(body.old_password, current_user.hashed_password):
        return {"result":"Fail","message":"user name or old password wrong"},200
    else:
        fake_users_db[current_user.username]["hashed_password"] = await get_password_hash(body.new_password)
        config["users"] = fake_users_db
        with open('config.json', 'w') as fp:
            json.dump(config, fp)
        return {"result":"Success","message":""},200


@app.get("/getCameras")
async def getCameras(current_user: User = Depends(get_current_active_user)):
    return {"result":list_camera},200


@app.post("/addCamera")
async def addCamera(body:AddCamera ,current_user: User = Depends(get_current_active_user)):
    global list_camera

    #if not await verify_password(body.password, current_user.hashed_password):
    #    return {"result":"Fail","message":"Password wrong! Please enter right password to add camera"},200

    if body.CameraID in list_camera.keys():
        return {"result":"Fail","message":body.CameraID +" already exist in database"},200
    else:
        list_camera[body.CameraID]={"status":body.status,"rstp_link":body.rstp_link}
        ##### create cv2 capture if status is 1:
        if body.status == "1":
            gst_str = ('rtspsrc location={} latency=0 ! queue max-size-buffers=1 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True').format(list_camera[body.CameraID]["rstp_link"])
            camera_active[body.CameraID] = cv2.VideoCapture(gst_str)
           # camera_active[body.CameraID] = cv2.VideoCapture(body.rstp_link)
        config["cameras"] = list_camera
        with open('config.json', 'w') as fp:
            json.dump(config, fp)
        return {"result":"Success","message":"Successfully add "+body.CameraID +" to database"},200

@app.post("/editCamera")
async def editCamera(body:AddCamera ,current_user: User = Depends(get_current_active_user)):
    global list_camera

   # if not await verify_password(body.password, current_user.hashed_password):
    #    return {"result":"Fail","message":"Password wrong! Please enter right password to add camera"},200

    if body.CameraID not in list_camera.keys():
        return {"result":"Fail","message":body.CameraID +" does not exist in database"},200
    else:
        for key in list_camera.keys():
            if body.rstp_link == list_camera[key]["rstp_link"] and body.CameraID != key:
                return {"result":"Fail","message":"your rstp link already assigned to "+key},200
        list_camera[body.CameraID]={"status":body.status,"rstp_link":body.rstp_link}
        ##### connect Camera if activate
        if body.status == "1":
            gst_str = ('rtspsrc location={} latency=0 ! queue max-size-buffers=1 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True').format(list_camera[body.CameraID]["rstp_link"])
            camera_active[body.CameraID] = cv2.VideoCapture(gst_str)
            #camera_active[body.CameraID] = cv2.VideoCapture(body.rstp_link)
        elif body.status == "0":
            camera_active[body.CameraID] = None
        config["cameras"] = list_camera
        with open('config.json', 'w') as fp:
            json.dump(config, fp)
        return {"result":"Success","message":"Successfully edit "+body.CameraID +""},200

@app.post("/testCamera")
async def testCamera(body:TestCamera ,current_user: User = Depends(get_current_active_user)):
    if body.CameraID not in list_camera.keys():
        return {"result":"Fail","message":body.CameraID +" does not exist in database"},200
    else:
        vid = cv2.VideoCapture(list_camera[body.CameraID]["rstp_link"]) 
        time.sleep(1)
        for i in range(5):
            ret, frame = vid.read() 
            if ret != None and np.all(frame != None) == True:
                return {"result":"Success","message":"Successfully connect "+body.CameraID},200
            time.sleep(1)
        return {"result":"Fail","message":"Fail to connect "+body.CameraID},200     

@app.post("/deleteCamera")
async def deleteCamera(body: DeleteCamera ,current_user: User = Depends(get_current_active_user)):
    global list_camera
    global camera_active
   # if not await verify_password(body.password, current_user.hashed_password):
   #     return {"result":"Fail","message":"Password wrong! Please enter right password to add camera"},200
    ### delete cv2 capture if cameraID in active
    if body.CameraID in camera_active.keys():
        camera_active = removekey(camera_active, body.CameraID)

    if body.CameraID not in list_camera.keys():
        return {"result":"Fail","message":body.CameraID +" does not exist in database"},200
    else:
        list_camera= removekey(list_camera, body.CameraID)
        config["cameras"] = list_camera
        with open('config.json', 'w') as fp:
            json.dump(config, fp)
        return {"result":"Success","message":"Successfully delete "+ body.CameraID +" from database"},200


@app.post("/checkUpdate")
async def checkUpdate(body : Update ,current_user: User = Depends(get_current_active_user)):

    if not await verify_password(body.password, current_user.hashed_password):
        return {"result":"Fail","message":"Password wrong"}, 200
    else:
        with open('config.json', 'r') as fp:
            config = json.load(fp)
            version = config["version"]
        x = None
        with eventlet.Timeout(10):
            data = {'version': version}
            x = requests.post(url = "http://14.177.239.164:9090/VerifyInformation", json = data)
        if x != None:
            final = x.json()
            final["result"] = "Success"
        else:
            final = {"result":"Fail","message":"request timeout" }
        return final, 200
        
@app.post("/installUpdate")
async def installUpdate(body : Update ,current_user: User = Depends(get_current_active_user),version: str = version):
    if current_user.username != body.username or not await verify_password(body.password, current_user.hashed_password):
        return {"result":"Fail","message":"user name or password wrong"},200
    else:
        
        ###execute update
        
        return final,200        

@app.post("/reboot")
async def reboot(current_user: User = Depends(get_current_active_user)):
        os.system("reboot")
        
        ###execute update
        
        return "OK",200 
