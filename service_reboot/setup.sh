#!/bin/bash
rm $HOME/nddung105/Code_api/ONNX_model/*onnx
rm $HOME/nddung105/Code_api/TRT_model/*trt

unzip -j $HOME/nddung105/Code_api/ONNX_model/*zip -d $HOME/nddung105/Code_api/ONNX_model/
rm $HOME/nddung105/Code_api/ONNX_model/*zip

run_python=/home/techpro/anaconda3/envs/thuannh/bin/python
uvicorn=/home/techpro/anaconda3/envs/thuannh/bin/uvicorn
export PYTHONPATH=/home/techpro/anaconda3/envs/thuannh/lib/python3.6

cd $HOME/nddung105/Code_api/ONNX_model
$run_python ocr_.py
$run_python onnx_to_tensorrt.py
cd $HOME/nddung105/Code_api/

mv $HOME/nddung105/Code_api/ONNX_model/*trt $HOME/nddung105/Code_api/TRT_model

$uvicorn main_v2:app --port 8000 --host 0.0.0.0 & \
$uvicorn server_api:app --host 0.0.0.0 --port 9090
