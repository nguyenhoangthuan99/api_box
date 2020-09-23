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
    status: Optional[str] = 1
    password : str

class DeleteCamera(BaseModel):
    CameraID : str
    password : str

class EditCamera(BaseModel):
    CameraID : str
    rstp_link: Optional[str] = None
    status: Optional[str] = 1
    password : str

class Update(BaseModel):
    username :str
    new_password : str

class TestCamera(BaseModel):
    CameraID: str 


