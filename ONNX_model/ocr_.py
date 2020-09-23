#!/usr/bin/env python2
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
import time
#from yolov3_to_onnx import download_file
#from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
import itertools
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
import cv2
TRT_LOGGER = trt.Logger()
if True:
        letters  =  " 1234567890QWERTYIUPLKJHGFDASZXCVBNM-."
        index = 0
        char2index = {}
        
        for i in letters:
            char2index[i] = index
            index += 1
        index2char = {}

        for u,v in char2index.items():
            index2char[v] = u
if True:
    def labels_to_text(labels):
        re = ""
        for i in labels:
           try:
               m = index2char[i]
               re += m
           except:
              continue
        return re 
    def decode_batch(out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = labels_to_text(out_best)
            ret.append(outstr)
        return ret
def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:#
            builder.max_workspace_size = 1 << 29 # 256MiB
            builder.max_batch_size = 1
            builder.fp16_mode = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            print(network.get_input(0).shape)
            network.get_input(0).shape = [1, 32, 128,1]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    
    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'ocr_10.onnx'
    engine_file_path = "ocr_noLSTM_fp16.trt"
    # Download a dog image and save it to the following file path:
    image = np.zeros((32,128))
    
    # Output shapes expected by the post-processor
    output_shapes = [(1, 32, 39),]#, (1, 255, 76, 76)]
    # Do inference with TensorRT
    trt_outputs = []
   # image = cv2.imread("144.jpg",0)
    image = np.expand_dims(image,0)
    image = np.expand_dims(image/255.,-1)
    image = np.array(image, dtype=np.float32, order='C')
    start = time.time()
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
      inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    for i in range(1):
        
        # Do inference
        
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs_[0].host = image
        trt_outputs = common.do_inference_v2(context_, bindings=bindings_, inputs=inputs_, outputs=outputs_, stream=stream_)
    print(len(trt_outputs))
    print(time.time()-start)
    
    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    return decode_batch(trt_outputs[0])
    
"""
#f = open("/media/thuan/New Volume/Data/val_ocr.txt")
#Line = f.readlines()
onnx_file_path1 = 'ocr_10.onnx'
engine_file_path1 = "ocr_noLSTM.trt"
engine_ocr = get_engine(onnx_file_path1, engine_file_path1) 
inputs_, outputs_, bindings_, stream_ = common.allocate_buffers(engine_ocr)
context_ = engine_ocr.create_execution_context() 
#imgs = glob.glob("/media/thuan/New Volume/Data/*")

total  = 0
true = 0
print(len(Line))
print(Line[0])
imgs = []
for line in Line:
  try:
    output_shapes = [(1, 32, 39),]#, (1, 255, 76, 76)]
    # Do inference with TensorRT
    trt_outputs = []
    line = line.replace("\n","")
    name = line.split(" ")[0]
    label = line.split(" ")[1]
    image = cv2.imread("/media/thuan/New Volume/Data/OCR_val/"+name,0)
    #image = cv2.resize(image,(32,12))
    image = np.expand_dims(image,0)
    image = np.expand_dims(image/255.,-1)
    image = np.array(image, dtype=np.float32, order='C')
    inputs[0].host = image
    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    pred = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)][0]
    
    bpred = decode_batch(pred)[0]
    label = label.replace(".","")
    label = label.replace("-","")
    bpred = bpred.replace(".","")
    bpred = bpred.replace("-","")
    total += 1
    if label == bpred: 
        true+=1
    ##else:
       # print(bpred, label)
  except: 
    print(line)
    continue   
print(true/total)
"""
if __name__ == '__main__':
    main()

