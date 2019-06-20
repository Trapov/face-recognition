import cv2
import numpy as np

im = cv2.imread('Aisha_Hinds_35703.png')

fe = cv2.dnn.readNetFromCaffe(prototxt='Resources_spherenet36.prototxt', caffeModel='FE.caffemodel')
size = 128

blob = cv2.dnn.blobFromImage(im, 0.00390625,
        (size, size), (104.0, 177.0, 123.0))

fe.setInput(blob)

results = fe.forward('norm2')

print(results)