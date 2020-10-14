#!/bin/bash

export DEVICE="jetson"
  
uvicorn main_v4:app --host 0.0.0.0 --port 8000
