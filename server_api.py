from fastapi import Depends, FastAPI, HTTPException, status
from item import *
import datetime
import json
import base64
import numpy as np
from PIL import Image
import os, io, cv2, time

app = FastAPI()

with open('version.json', 'r') as fp:
    version = json.load(fp)
    last_version = max(version['version'], key=lambda ev: ev['version'])

new_version = int(last_version["version"].replace('.', ''))
date = last_version["date"]
link_zip = last_version["link_zip"]
description = last_version["description"]
size = last_version["size"]
md5 = last_version["md5"]

@app.post("/VerifyInformation")
def VerifyInformation(body: VerifyInformation):
    client_version = body.version
    client_version = int(client_version.replace('.', ''))
    if new_version <= client_version:
        print(new_version, client_version)
        return {"result": "there is no new version"}
    else:
        return {"version": last_version["version"],"date":date, "link_zip":link_zip, 
                 "description":description,"size": size, "md5":md5}

@app.post("/updateVersion")
def updateVersion(body: updateVersion):
    with open('version.json', 'r') as fp:
        version = json.load(fp)
    version["AIBOX-CLIENT"][body.AIBOX_ID] = {}
    version["AIBOX-CLIENT"][body.AIBOX_ID]["date_update"] = str(datetime.date.today())
    version["AIBOX-CLIENT"][body.AIBOX_ID]["status"] = body.status
    version["AIBOX-CLIENT"][body.AIBOX_ID]["new_version"] = body.new_version

    with open('version.json', 'w') as fp:
        json.dump(version, fp)

    return {"message":"OK"}, 200


@app.post("/saveWrongImage")
def saveWrongImage(body: saveWrongImage):
    today = str(datetime.date.today())
    if not os.path.exists("wrong_image/input/"+today):
        os.makedirs("wrong_image/input/"+today)
    if not os.path.exists("wrong_image/output/"+today):
        os.makedirs("wrong_image/output/"+today)
    
    name = str(round(time.time() * 1000))

    fileb = base64.b64decode(body.input_image)
    img = np.array(Image.open(io.BytesIO(fileb)).convert("RGB"))
    tmp = Image.fromarray(np.uint8(img))
    tmp.save("wrong_image/input/"+today+"/"+name+".jpg", 'jpeg')

    fileb = base64.b64decode(body.output_image)
    img = np.array(Image.open(io.BytesIO(fileb)).convert("RGB"))
    tmp = Image.fromarray(np.uint8(img))
    tmp.save("wrong_image/output/"+today+"/"+name+".jpg", 'jpeg')

    with open('wrong_image/log_wrong_image.json', 'r') as fp:
        try:
            wrong_image = json.load(fp)
        except:
            wrong_image = {}
        wrong_image[name]={"time": body.time, "camera_id": body.camera_id, "AIBOX_ID": body.AIBOX_ID}

    with open('wrong_image/log_wrong_image.json', 'w') as fp:
        json.dump(wrong_image, fp)
    return "OK", 200
    

# ===================================
@app.post("/recognition")
async def recognition(body:ImagE):
    fileb = base64.b64decode(body.image)
    img = np.array(Image.open(io.BytesIO(fileb)).convert("RGB"))
    
    tmp = Image.fromarray(np.uint8(img))
    tmp.save("/opt/lampp/htdocs/recognition.jpg", 'jpeg')

    return "OK", 200

@app.post("/unrecognition")
async def unrecognition(body:ImagE):
    fileb = base64.b64decode(body.image)
    img = np.array(Image.open(io.BytesIO(fileb)).convert("RGB"))
    
    tmp = Image.fromarray(np.uint8(img))
    tmp.save("/opt/lampp/htdocs/unrecognition.jpg", 'jpeg')

    return "OK", 200