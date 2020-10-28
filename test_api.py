import os, time, json, io
import datetime, base64
import numpy as np
from PIL import Image
from threading import Thread

def process_id(id):
    """process a single ID"""
    data = os.popen('curl -X POST "http://14.177.239.164:8001/decode?CameraID=' + str(id) + "--" + '" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "fileb=@33.jpg;type=image/jpeg"').read()
    return data

def process_range(id_range, store=None):
    """process a number of ids, storing the results in a dict"""
    if store is None:
        store = {}
    for id in id_range:
        store[id] = process_id(id)
    return store

def threaded_process_range(nthreads, id_range):
    """process the id range in a specified number of threads"""
    store = {}
    threads = []
    # create the threads
    for i in range(nthreads):
        ids = id_range[i::nthreads]
        t = Thread(target=process_range, args=(ids,store))
        threads.append(t)

    # start the threads
    start = time.time()
    [ t.start() for t in threads ]
    # wait for the threads to finish
    [ t.join() for t in threads ]
    api_time = time.time() - start
    print('--------', api_time)
    return store

#################################

# Setup các thread
id_range = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
nthreads = len(id_range)

# Call API đồng thời
start_ = time.time()
threaded_process_range(nthreads, id_range)
api_time_ = time.time() - start_
print('+++++++++++', api_time_)

# Call API tuần tự
start_ = time.time()
for id in id_range:
  process_id(id)
api_time_ = time.time() - start_
print('+++++++++++', api_time_)
