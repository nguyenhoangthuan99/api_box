from fastapi import APIRouter, HTTPException
from fastapi import  HTTPException, Form,File,UploadFile, WebSocket,Depends, status
from datetime import datetime, timedelta
from typing import Optional
import pickle, json
import requests
import eventlet

fp = open('config.json', 'r')
version = json.load(fp)["version"]
fp.close()
with open('config.json', 'r') as fp:
    config = json.load(fp)

fake_users_db = config["users"]

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

@router.post("/checkUpdate")
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

@router.post("/installUpdate")
async def installUpdate(body : Update ,current_user: User = Depends(get_current_active_user),version: str = version):
    if not await verify_password(body.password, current_user.hashed_password):
        return {"result":"Fail","message":"Password wrong"}, 200
    else:
        x = None
        with eventlet.Timeout(10):
            x = requests.post(url="http://14.177.239.164:9090/VerifyInformation",json={"password":body.password,"version":version})
        if x != None:
            final = x.json()
            if not final.get("result"):
                version = final["version"]
                with open('config.json', 'r') as fp:
                    config = json.load(fp)
                config["version"] = version
                config["update_require"] = True
                config["md5"] = final["md5"]
                # config["date_update"] = final["date"]
                with open("config.json","w") as fp:
                    json.dump(config,fp)
                link_zip = final["link_zip"]
                os.system(
                    '''wget {} -c -O ./ONNX_model_check/ONNX_model.zip'''.format(link_zip))
                os.system("echo 1|sudo -S reboot")

