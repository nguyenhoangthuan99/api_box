#!/bin/bash
export PATH=/home/techpro/.nvm/versions/node/v12.18.4/bin:usr/local/cuda/bin/:/home/techpro/anaconda3/bin:/home/techpro/anaconda3/condabin:/usr/local/cuda/bin:/home/techpro/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/::/home/techpro/TensorRT-6.0.1.5/lib:/home/techpro/TensorRT-7.0.0.11/lib

conda deactivate

rm $HOME/nddung105/Code_api/ONNX_model/*onnx
rm $HOME/nddung105/Code_api/TRT_model/*trt

unzip -j $HOME/nddung105/Code_api/ONNX_model/*zip -d $HOME/nddung105/Code_api/ONNX_model/
rm $HOME/nddung105/Code_api/ONNX_model/*zip

cd $HOME/nddung105/Code_api/ONNX_model
python3 ocr_.py
python3 onnx_to_tensorrt.py
cd $HOME/nddung105/Code_api/

mv $HOME/nddung105/Code_api/ONNX_model/*trt $HOME/nddung105/Code_api/TRT_model

uvicorn main_v2:app --port 8000 --host 0.0.0.0 & \
uvicorn server_api:app --host 0.0.0.0 --port 9090
