import asyncio
import datetime
import random
import time as t
import websockets
import numpy as np
import cv2
import json
import base64

from imutils import face_utils
import imutils
import dlib

class AlignerCPUDlib:
    def __init__(self):
        self.__predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def getKeypoints(self, image, box):
        rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = self.__predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        return shape

class AlignerCPUCaffe:
    def __init__(self):
        self.__net = cv2.dnn.readNetFromCaffe('alignment.prototxt', 'alignment.caffemodel')
        self.scale = 0.00390625

    def resize_with_padding(self, im):
        desired_size = 64
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(im, (new_size[1], new_size[0]))

        #cv2.imwrite('before.jpeg', im)

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        
        #cv2.imwrite('after.jpeg', new_im)

        return new_im

    def align(self, image: np.ndarray):
        image = image.astype('float32', copy=False) 
        orig_shape = image.shape[:2]
        size = 64
        coef_y, coef_x = size/orig_shape[0], size/orig_shape[1]

        print("COEFF => ", coef_y, coef_x)

        resized_image = self.resize_with_padding(image)
        resized_image = np.rot90(resized_image)
        resized_image *= self.scale

        blob = cv2.dnn.blobFromImage(resized_image, 1.0,
                (size, size), (104.0, 177.0, 123.0))

        self.__net.setInput(blob)
        result = self.__net.forward('fc5')

        out_arr = []

        y = False

        for point in result[0]:
            out_arr.append(point * 64 * coef_y if y else point * 64 * coef_x)
            y = not y

        return out_arr
