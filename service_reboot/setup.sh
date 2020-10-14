#!/bin/bash

export PATH=/home/xuanhung/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda/bin
export LD_LIBRARY_PATH=:/usr/local/cuda/lib64
export DEVICE="jetson"

#conda deactivate

cd $HOME/Code_api_edit/Code_api/

update_require=$(cat config.json | jq --raw-output '.update_require')
md5=$(cat config.json | jq '.md5')

if [ $update_require == true ]
then
	md5_check=$(md5sum $HOME/Code_api_edit/Code_api/ONNX_model_check/ONNX_model.zip|cut -f 1 -d " ")
	if [ $md5 == $md5_check]
	then
		unzip -j $HOME/Code_api_edit/Code_api/ONNX_model_check/*zip -d $HOME/Code_api_edit/Code_api/ONNX_model_check/
		cd $HOME/Code_api_edit/Code_api/ONNX_model_check
		python3 ocr_.py
		python3 onnx_to_tensorrt.py
		number_trt=$(find *trt | wc -l)
		if [ $number_trt == 2 ]
		then
			mv $HOME/Code_api_edit/Code_api/ONNX_model_check/*trt $HOME/Code_api_edit/Code_api/TRT_model
			rm $HOME/Code_api_edit/Code_api/ONNX_model_check/*zip
			mv $HOME/Code_api_edit/Code_api/ONNX_model_check/*onnx $HOME/Code_api_edit/Code_api/ONNX_model
			mv $HOME/Code_api_edit/Code_api/ONNX_model_check/ocr_.py $HOME/Code_api_edit/Code_api/ONNX_model
			mv $HOME/Code_api_edit/Code_api/ONNX_model_check/onnx_to_tensorrt.py $HOME/Code_api_edit/Code_api/ONNX_model
			mv $HOME/Code_api_edit/Code_api/ONNX_model_check/main_v3.py $HOME/Code_api_edit/Code_api/
			cd $HOME/Code_api_edit/Code_api
			jq '.update_require=false' config.json > config.json.tmp && cp config.json.tmp config.json
			sleep 30
			uvicorn main_v4:app --port 8001 --host 0.0.0.0
		else
			rm -r $HOME/Code_api_edit/Code_api/ONNX_model_check/*
			cd $HOME/Code_api_edit/Code_api
			jq '.update_require=false' config.json > config.json.tmp && cp config.json.tmp config.json
			sleep 30
			uvicorn main_v4:app --port 8001 --host 0.0.0.0
		fi
	else
		rm -r $HOME/Code_api_edit/Code_api/ONNX_model_check/*
		cd $HOME/Code_api_edit/Code_api
		jq '.update_require=false' config.json > config.json.tmp && cp config.json.tmp config.json
		sleep 30
		uvicorn main_v4:app --port 8001 --host 0.0.0.0
	fi
else
	# rm -r $HOME/Code_api_edit/Code_api/ONNX_model_check/*
	cd $HOME/Code_api_edit/Code_api
	jq '.update_require=false' config.json > config.json.tmp && cp config.json.tmp config.json
	sleep 30
	uvicorn main_v4:app --port 8001 --host 0.0.0.0
fi
#uvicorn server_api:app --host 0.0.0.0 --port 9090
