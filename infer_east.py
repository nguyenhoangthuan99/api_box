import numpy as np
import os
from data_processor import * #restore_rectangle
import lanms
from post_process_box import *
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
import time
import common

TRT_LOGGER = trt.Logger()




def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.15, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]
class EAST:
    def __init__(self,model_path_small,model_path_large):
        with open(model_path_small, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine_s =  runtime.deserialize_cuda_engine(f.read())
        self.inputs_s, self.outputs_s, self.bindings_s, self.stream_s = common.allocate_buffers(self.engine_s)
        self.context_s = self.engine_s.create_execution_context() 
 
        with open(model_path_large, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine_l =  runtime.deserialize_cuda_engine(f.read())
        self.inputs_l, self.outputs_l, self.bindings_l, self.stream_l = common.allocate_buffers(self.engine_l)
        self.context_l = self.engine_l.create_execution_context() 
   
    def predict(self,batch_x):
        result = []
        img = batch_x
        h,w,_ = img.shape
        if abs(h-288) + abs(w-352) < abs(h-704) + abs(w-1280):
            flag = 0
            output_shapes = [(1, 72, 88, 1), (1, 72, 88, 5)]
            inputs, outputs, bindings, stream,context = self.inputs_s, self.outputs_s, self.bindings_s, self.stream_s,self.context_s
        else:
            flag = 1
            output_shapes = [(1, 176, 320, 1), (1, 176, 320, 5)]
            inputs, outputs, bindings, stream,context = self.inputs_l, self.outputs_l, self.bindings_l, self.stream_l,self.context_l

        start_time = time.time()
        #img,shift_h, shift_w = pad_image(img,512,False)
        if flag == 0 :
            batch_x = cv2.resize(batch_x,(352,288))
        else: 
            batch_x = cv2.resize(batch_x,(1280,704))
        img = batch_x.copy()
        img_resized, (ratio_h, ratio_w) = resize_image(batch_x)
        
        img_resized = (img_resized / 127.5) - 1
        img_resized = img_resized[np.newaxis, :, :, :]

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()

        inputs[0].host = np.array(img_resized , dtype=np.float32, order='C')
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

        score_map, geo_map = trt_outputs[0],trt_outputs[1]
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score_map, geo_map=geo_map, timer=timer)
        print(' net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
             timer['net']*1000, timer['restore']*1000, timer['nms']*1000))
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        print('[timing] {}'.format(duration))
        
        if boxes is not None:
          

           
            for box in boxes:
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    result.append([ box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]])
                   
                    #cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)

        
        result,bbox = logic(result,img)
        """
        for i in range(np.array(bbox).shape[0]):
          #  print(bbox[i].shape)
            tmp = bbox[i][2].copy()
            bbox[i][2] = bbox[i][3]
            bbox[i][3] = tmp
            cv2.polylines(img[:, :, ::-1], [bbox[i].astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)

       # cv2.imwrite("sample.jpg", img[:, :, ::-1])
        """
        return result,bbox,img

import time
east = EAST("./TRT_model/east_288_352_fp16.trt","./TRT_model/east_python_704_1280_fp16.trt")
"""
import cv2  #/tf_model_96_0.1.pb
img = cv2.imread("test_/00000000219000000_1227.jpg")[:,:,::-1]

#img = cv2.resize(img,(1280,704))

#img = np.expand_dims(img/255.,0)
#img = np.expand_dims(img,-1)
#img = tf.convert_to_tensor(img, dtype=tf.float32)
#img = tf.constant(img)
for i in range(2): east.predict(img)
start = time.time()
for i in range(1000): east.predict(img)
print(time.time()-start)
"""

