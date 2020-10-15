from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modules import cam_login_api
from modules import decode
from modules import update

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

app.include_router(cam_login_api.router)
app.include_router(decode.router)
app.include_router(update.router)
