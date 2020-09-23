import numpy as np
import os,time
import itertools
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import common
import cv2
TRT_LOGGER = trt.Logger()


class OCRDecoder:
    def __init__(self,model_path):
        
        self.model_path = model_path
      
        self.letters  = " 1234567890QWERTYIUPLKJHGFDASZXCVBNM-." 
        index = 0
        char2index = {}
        
        for i in self.letters:
            char2index[i] = index
            index += 1
        index2char = {}

        for u,v in char2index.items():
            index2char[v] = u
        self.char2index = char2index
        self.index2char = index2char
        with open(self.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine =  runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context() 
    def labels_to_text(self,labels):
        re = ""
        for i in labels:
           try:
               m = self.index2char[i]
               re += m
           except:
              continue
        return re 
    def decode_batch(self,out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = self.labels_to_text(out_best)
            ret.append(outstr)
        return ret
    def predict(self,batch_x):
        s = time.time()
        imgs2 = np.array(batch_x)
        imgs2 = np.expand_dims(imgs2/255.,-1)
        result = []
        output_shapes = [(1, 32, 39),]
        for i in range(imgs2.shape[0]):
            img = np.expand_dims(imgs2[i],0)

            trt_outputs = []
            img = np.array(img, dtype=np.float32, order='C')
            self.inputs[0].host = img
            trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
            preds = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)][0]
            out = self.decode_batch(preds)
            result += out
        print("ocr time: ",time.time()-s)
        return result
      
ocr = OCRDecoder("./TRT_model/ocr_noLSTM_fp16.trt")#eff_l2_OCR_97
"""
img = cv2.imread("text_box/1.jpg",0)#[:,:,::-1]

imgs = [cv2.resize(img,(128,32)),cv2.resize(img,(128,32)),cv2.resize(img,(128,32)),cv2.resize(img,(128,32))]
for i in range (1): print(ocr.predict(imgs))
"""





