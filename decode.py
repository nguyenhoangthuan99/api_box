import os,io, cv2
import numpy as np
import time,pickle
import base64
from PIL import Image
import requests,ast,json
from fastapi import APIRouter, HTTPException, Form,File,UploadFile, WebSocket,Depends, status
from collections import OrderedDict
from starlette.websockets import WebSocket, WebSocketDisconnect
from datetime import datetime, timedelta
from end2end import inference
from cam_login_api import list_camera, camera_active
from item import Decode2
router = APIRouter()

def d(x1,y1,x2,y2):
   return ((x1-x2)**2  +(y1-y2)**2)**0.5

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

async def process(img,CameraID,final):
        
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


        return final

@router.post("/stream_decode_gstreamer")
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
        return await process(img,CameraID,final)
        
@router.post("/stream_decode")
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
        return await process(img,CameraID,final)

@router.post("/decode2")
async def decodes2(body:Decode2):

        final ={}
        final["received_time"] = time.strftime("%d-%m-%Y %H:%M:%S")

        fileb = base64.b64decode(body.image)
        img = np.array(Image.open(io.BytesIO(fileb)).convert("RGB"))
        return await process(img,body.CameraID)


@router.post("/decode")
async def decodes(CameraID: str , fileb: bytes = File(...),):
        final ={}
        final["received_time"] = time.strftime("%d-%m-%Y %H:%M:%S")
       # if not os.path.exists("logs_txt/"+time.strftime("%d-%m-%Y")):
       #     os.mkdir("logs_txt/"+time.strftime("%d-%m-%Y"))
        
        img = np.array(Image.open(io.BytesIO(fileb)).convert("RGB"))
        return await process(img,CameraID,final)


@router.websocket("/plate-recognition")
async def websocket_endpoint(websocket: WebSocket):
    await notifier.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
           # await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        await notifier.remove(websocket)
   

@router.on_event("startup")
async def startup():
    # Prime the push notification generator
    await notifier.generator.asend(None)
