from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modules import cam_login_api
from modules import decode
from modules import update
import requests
import json

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


@app.on_event("startup")
async def startup_event():
    fp = open('config.json', 'r')
    data = json.load(fp)
    version = data["version"]
    update_success = data["update_success"] 
    fp.close()
    if update_success:
        requests.post(url="http://14.177.239.164:9090/updateVersion",json={"AIBOX_ID":"aibox_id_1","status":"","new_version":version})
    data["update_success"] = False
    with open("config.json","w") as fp:
        json.dump(data,fp)  

app.include_router(cam_login_api.router)
app.include_router(decode.router)
app.include_router(update.router)
