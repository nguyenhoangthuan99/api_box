from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
import werkzeug
from flask import  jsonify, send_from_directory
import cv2
from flask_cors import CORS
import os
import numpy as np
from functools import wraps
import tensorflow as tf
app = Flask(__name__)
api = Api(app)
ALLOWED_EXTENSIONS = ["jpg", "png"]
import time
import base64
import io
from OCRDecode import ocr

class OCRAPI(Resource):
    def post(self):
        #try:
            #parser = reqparse.RequestParser()
            #parser.add_argument('image',type=str)
            #self.args = parser.parse_args()
        #except:
            #self.error_response(-19403)
            return self.predict()

    def predict(self):
       # print(self.args["image"])
       # r = base64.decodebytes(self.args["image"].encode("utf-8"))
        
       # q = np.frombuffer(r, dtype=np.float32)

        data = request.json
       
        q = np.array(data['image'])
        res = ocr.predict(q)
       
        return jsonify({'result': res})


api.add_resource(OCRAPI, "/ocr/predict")

if __name__ == '__main__':
   
    app.run(port = 5000)


