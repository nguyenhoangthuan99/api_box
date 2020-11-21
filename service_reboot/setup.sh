#!/bin/bash

# Config path and library
export PATH=/home/mic-710iva/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda/bin
export LD_LIBRARY_PATH=:/usr/local/cuda/lib64
export DEVICE="jetson"

#conda deactivate

cd $HOME/api_box/

# Get update_require and md5 from config.json file
update_require=$(cat config.json | jq --raw-output '.update_require')
md5=$(cat config.json | jq '.md5')

# If require update
if [ $update_require == true ]; then

	# Check md5
	md5_check=$(md5sum $HOME/api_box/ONNX_model_check/ONNX_model.zip | cut -f 1 -d " ")
	md5_check='"'$md5_check'"'
	if [ $md5 == $md5_check ]; then

		unzip -j $HOME/api_box/ONNX_model_check/*zip -d $HOME/api_box/ONNX_model_check/
		cd $HOME/api_box/ONNX_model_check

		# Have .sh fileß
		if [ -f $HOME/api_box/ONNX_model_check/service_reboot/setup.sh ]; then
			chmod +x $HOME/api_box/ONNX_model_check/service_reboot/setup.sh
			$HOME/api_box/ONNX_model_check/service_reboot/setup.sh

		else
			# Have file ONNX
			if ls $HOME/api_box/ONNX_model_check/*.onnx &>/dev/null; then
				python3 ocr_.py
				python3 onnx_to_tensorrt.py
				number_trt=$(find *trt | wc -l)

				# If convert success, have 2 file trt
				if [ $number_trt == 2 ]; then
					mv $HOME/api_box/ONNX_model_check/*trt $HOME/api_box/TRT_model
					rm $HOME/api_box/ONNX_model_check/*zip
					mv $HOME/api_box/ONNX_model_check/* $HOME/api_box/
					cd $HOME/api_box
					jq '.update_require=false' config.json >config.json.tmp && cp config.json.tmp config.json
					jq '.update_success=true' config.json >config.json.tmp && cp config.json.tmp config.json
					sleep 30
					uvicorn main_v4:app --port 8000 --host 0.0.0.0
				fi

			# Else not have file ONNX
			else
				rm $HOME/api_box/ONNX_model_check/*zip
				mv $HOME/api_box/ONNX_model_check/* $HOME/api_box/
				cd $HOME/api_box
				jq '.update_require=false' config.json >config.json.tmp && cp config.json.tmp config.json
				jq '.update_success=true' config.json >config.json.tmp && cp config.json.tmp config.json
				sleep 30
				uvicorn main_v4:app --port 8000 --host 0.0.0.0
			fi
		fi

	# Else md5
	else
		rm -r $HOME/api_box/ONNX_model_check/*
		cd $HOME/api_box
		jq '.update_require=false' config.json >config.json.tmp && cp config.json.tmp config.json
		sleep 30
		uvicorn main_v4:app --port 8000 --host 0.0.0.0
	fi

# Else not update
else
	# rm -r $HOME/api_box/ONNX_model_check/*
	cd $HOME/api_box
	jq '.update_require=false' config.json >config.json.tmp && cp config.json.tmp config.json
	sleep 30
	uvicorn main_v4:app --port 8000 --host 0.0.0.0
fi
