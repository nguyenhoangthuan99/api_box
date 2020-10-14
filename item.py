from typing import Optional
from pydantic import BaseModel
from fastapi import File

class ChangePassword(BaseModel):
    username :str
    old_password : str
    new_password : str
    
class AddCamera(BaseModel):
    CameraID : str
    rstp_link: str
    status: Optional[str] = "1"
   # password : str

class DeleteCamera(BaseModel):
    CameraID : str
   # password : str

class EditCamera(BaseModel):
    CameraID : str
    rstp_link: Optional[str] = None
    status: Optional[str] = "1"
   # password : str

class Update(BaseModel):
    password : str

class TestCamera(BaseModel):
    CameraID: str 

class Decode2(BaseModel):
    CameraID : str
    image: str

class ImagE(BaseModel):
    image: str

class checkUpdate(BaseModel):
    username : str
    password: str
    version:str

class VerifyInformation(BaseModel):
    version: str

class updateVersion(BaseModel):
    AIBOX_ID : str
    new_version: str
    status: str

class saveWrongImage(BaseModel):
    input_image : str
    output_image: str
    time: str
    camera_id: str
    AIBOX_ID: str