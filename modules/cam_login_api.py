from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from typing import Optional
import pickle, json
import base64,cv2


with open('config.json', 'r') as fp:
    config = json.load(fp)

fake_users_db = config["users"]
list_camera = config["cameras"]

camera_active = {}

cameras_ = [list_camera[i]["name"] for i in range(len(list_camera))]
for key in cameras_:
    if list_camera[cameras_.index(key)]["status"] == "1":
        print("CAPTURING ",list_camera[cameras_.index(key)]["rstp_link"])
        gst_str = ('rtspsrc location={} latency=0 ! queue max-size-buffers=1 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True').format(list_camera[cameras_.index(key)]["rstp_link"])
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
from fastapi import  HTTPException, Form,File,UploadFile, WebSocket,Depends, status
from collections import OrderedDict
from starlette.websockets import WebSocket, WebSocketDisconnect

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from item import *


router = APIRouter()

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


@router.post("/login", response_model=Token)
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


@router.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@router.get("/getVersion")
async def getVersion():
    fp = open('config.json', 'r')
    version = json.load(fp)["version"]
    fp.close()
    return version

@router.post("/changePassword")
async def changePassword(body: ChangePassword ,current_user: User = Depends(get_current_active_user)):
    if current_user.username != body.username or not await verify_password(body.old_password, current_user.hashed_password):
        return {"result":"Fail","message":"user name or old password wrong"},200
    else:
        fake_users_db[current_user.username]["hashed_password"] = await get_password_hash(body.new_password)
        config["users"] = fake_users_db
        with open('config.json', 'w') as fp:
            json.dump(config, fp)
        return {"result":"Success","message":""},200


@router.get("/getCameras")
async def getCameras(current_user: User = Depends(get_current_active_user)):
    return {"result":list_camera},200


@router.post("/addCamera")
async def addCamera(body:AddCamera ,current_user: User = Depends(get_current_active_user)):
    global list_camera

    #if not await verify_password(body.password, current_user.hashed_password):
    #    return {"result":"Fail","message":"Password wrong! Please enter right password to add camera"},200
    cameras_ = [list_camera[i]["name"] for i in range(len(list_camera))]
    if body.CameraID in cameras_:
        return {"result":"Fail","message":body.CameraID +" already exist in database"},200
    else:
        list_camera.append({"name": body.CameraID,"status":body.status,"rstp_link":body.rstp_link})
        ##### create cv2 capture if status is 1:
        if body.status == "1":
            gst_str = ('rtspsrc location={} latency=0 ! queue max-size-buffers=1 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True').format(list_camera[len(list_camera)-1]["rstp_link"])
            camera_active[body.CameraID] = cv2.VideoCapture(gst_str)
           # camera_active[body.CameraID] = cv2.VideoCapture(body.rstp_link)
        config["cameras"] = list_camera
        with open('config.json', 'w') as fp:
            json.dump(config, fp)
        return {"result":"Success","message":"Successfully add "+body.CameraID +" to database"},200

@router.post("/editCamera")
async def editCamera(body:AddCamera ,current_user: User = Depends(get_current_active_user)):
    global list_camera

   # if not await verify_password(body.password, current_user.hashed_password):
    #    return {"result":"Fail","message":"Password wrong! Please enter right password to add camera"},200

    cameras_ = [list_camera[i]["name"] for i in range(len(list_camera))]
    if body.CameraID not in cameras_:
        return {"result":"Fail","message":body.CameraID +" does not exist in database"},200
    else:
        for key in range(len(list_camera)):
            if (body.rstp_link == list_camera[key]["rstp_link"]) and (body.status == list_camera[key]["status"]):
                return {"result":"Fail","message":"your data already assigned to "+str(key)},200
        list_camera[cameras_.index(body.CameraID)]={"name": body.CameraID,"status":body.status,"rstp_link":body.rstp_link}
        ##### connect Camera if activate
        if body.status == "1":
            gst_str = ('rtspsrc location={} latency=0 ! queue max-size-buffers=1 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True').format(list_camera[cameras_.index(body.CameraID)]["rstp_link"])
            camera_active[body.CameraID] = cv2.VideoCapture(gst_str)
            #camera_active[body.CameraID] = cv2.VideoCapture(body.rstp_link)
        elif body.status == "0":
            camera_active[body.CameraID] = None
        config["cameras"] = list_camera
        with open('config.json', 'w') as fp:
            json.dump(config, fp)
        return {"result":"Success","message":"Successfully edit "+body.CameraID +""},200

@router.post("/testCamera")
async def testCamera(body:TestCamera ,current_user: User = Depends(get_current_active_user)):
    cameras_ = [list_camera[i]["name"] for i in range(len(list_camera))]
    if body.CameraID not in cameras_:
        return {"result":"Fail","message":body.CameraID +" does not exist in database"},200
    else:
        vid = cv2.VideoCapture(list_camera[cameras_.index(body.CameraID)]["rstp_link"]) 
        time.sleep(1)
        for i in range(5):
            ret, frame = vid.read() 
            if ret != None and np.all(frame != None) == True:
                return {"result":"Success","message":"Successfully connect "+body.CameraID},200
            time.sleep(1)
        return {"result":"Fail","message":"Fail to connect "+body.CameraID},200     

@router.post("/deleteCamera")
async def deleteCamera(body: DeleteCamera ,current_user: User = Depends(get_current_active_user)):
    global list_camera
    global camera_active
   # if not await verify_password(body.password, current_user.hashed_password):
   #     return {"result":"Fail","message":"Password wrong! Please enter right password to add camera"},200
    ### delete cv2 capture if cameraID in active
    cameras_ = [list_camera[i]["name"] for i in range(len(list_camera))]
    
    if body.CameraID in camera_active:
        camera_active = removekey(camera_active, body.CameraID)
    if body.CameraID not in cameras_:
        return {"result":"Fail","message":body.CameraID +" does not exist in database"},200
    else:
        del list_camera[cameras_.index(body.CameraID)]
        config["cameras"] = list_camera
        with open('config.json', 'w') as fp:
            json.dump(config, fp)
        return {"result":"Success","message":"Successfully delete "+ body.CameraID +" from database"},200








