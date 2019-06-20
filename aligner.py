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

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

class AlignerCPUDlib:
    def __init__(self):
        self.__predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def getKeypoints(self, image, box):
        rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = self.__predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        return shape

    def get_head_pose(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])

        _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

        reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                            dist_coeffs)

        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return reprojectdst, euler_angle
    
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
