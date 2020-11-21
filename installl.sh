
sudo nano /etc/pam.d/common-password >> minlen=1 >> passwd

sudo apt-get update
sudo apt-get install nano python-dev libjpeg-dev libjpeg8-dev libfreetype6-dev libgeos-dev libffi-dev

echo 'export CPATH=$CPATH:/usr/local/cuda-10.2/targets/aarch64-linux/include' >> ~/.bashrc 
echo 'export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/targets/aarch64-linux/lib' >> ~/.bashrc 
# sudo pip3 install pycuda --user
python3 -m pip install pycuda
sudo python3 -m pip install pycuda
sudo python3 -m pip install Pillow
sudo pip3 install fastapi uvicorn python-jose[cryptography] python-multipart shapely eventlet passlib[bcrypt]

wget http://14.177.239.164:7070/model_onnx.zip
unzip model_onnx.zip
cd model_onnx/ONNX_model
python3 ocr_.py
python3 onnx_to_tensorrt.py
mkdir ~/model_onnx/TRT_model
mv *.trt ~/model_onnx/TRT_model

export DEVICE="jetson"
echo 'export DEVICE="jetson"' >> ~/.bashrc 
uvicorn main_v4:app --host 0.0.0.0 --port 8000

gst-launch-1.0 rtspsrc location=rtsp://admin:Camera1234@171.244.236.152:554 latency=0 ! queue max-size-buffers=1 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=True 

gst-launch-1.0 -v playbin uri=rtsp://admin:Camera1234@171.244.236.152:554 uridecodebin0::source::latency=300

gst-launch-1.0 rtspsrc location=rtsp://admin:Camera1234@171.244.236.152:554 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! autovideosink
