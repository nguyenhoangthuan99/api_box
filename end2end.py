from infer_east import east
from OCRDecode import ocr
import numpy as np

class Inference:
    def __init__(self):
        self.ocr = ocr
        self.east = east
    def run(self,image):
        result,bbox,img = east.predict(image)
        ocr_res = ocr.predict(result)
        return ocr_res,bbox,img


inference = Inference()


