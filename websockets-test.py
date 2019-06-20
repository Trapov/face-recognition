import asyncio
import datetime
import random
import time as t
import websockets
import numpy as np
import cv2
import json
import base64

from aligner import AlignerCPUCaffe, AlignerCPUDlib, line_pairs

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
aligner_net = AlignerCPUDlib()

print(aligner_net)

def detect(image):
    (h, w) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)

    start = t.time()
    detections = net.forward()
    end = t.time()
    
    cropped_image = image

    print("[INFO] SSD took {:.6f} seconds".format(end - start))

    first_box = None
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.8:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            first_box = first_box or (startX, startY, endX, endY) 

            # cropped_image = image[startY:endY, startX:endX]
            # aligner_result = aligner_net.align(cropped_image)

            #cv2.imwrite('cropped.jpeg', cropped_image)

            # draw the bounding box of the face along with the associated
            # probability
            #y = startY - 10 if startY - 10 > 10 else startY + 10
            
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 5)

            keypoints = aligner_net.getKeypoints(image, box.astype("int"))

            reprojectdst, euler_angle = aligner_net.get_head_pose(keypoints)


            for (x, y) in keypoints:
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1) 

            for start, end in line_pairs:
                cv2.line(image, reprojectdst[start], reprojectdst[end], (0, 0, 255))

            print(f"HPE X:{euler_angle[0, 0]}, Y:{euler_angle[1, 0]}, Z:{euler_angle[2, 0]}")

            # cv2.putText(image, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (100, 20), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.75, (0, 0, 0), thickness=2)
            # cv2.putText(image, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.75, (0, 0, 0), thickness=2)
            # cv2.putText(image, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (100, 80), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.75, (0, 0, 0), thickness=2)

            # first_aligned =  aligner_result
            # cropped_h, cropped_w =  cropped_image.shape[:2]

            # cv2.circle(image, (int(first_aligned[0]+startX), int(first_aligned[1]+startY)), 2, (255, 0, 0), 2)
            # cv2.circle(image, (int(first_aligned[2]+startX), int(first_aligned[3]+startY)), 2, (255, 0, 0), 2)
            # cv2.circle(image, (int(first_aligned[4]+startX), int(first_aligned[5]+startY)), 2, (255, 0, 0), 2)
            # cv2.circle(image, (int(first_aligned[6]+startX), int(first_aligned[7]+startY)), 2, (255, 0, 0), 2)
            # cv2.circle(image, (int(first_aligned[8]+startX), int(first_aligned[9]+startY)), 2, (255, 0, 0), 2)
    x_center = first_box[0] + (first_box[2] - first_box[0])/2
    image = frame_crop(image, int(x_center))
    return image

cap = cv2.VideoCapture(0)

def frame_crop(frame, x_center):
    crop_width = 320
    crop_height = frame.shape[0]
    crop_x = x_center - int(320 / 2)
    if crop_x < 0:
        crop_x = 0
    if crop_x + crop_width > frame.shape[1]:
        crop_x = frame.shape[1] - crop_width
    try:
        cropped_frame = frame[0:crop_height, crop_x:crop_x+crop_width].copy()
    except:
        print("PIZDEC")
    return cropped_frame

async def time(websocket, path):
    while True:
        start_time = t.time()
        ret, img = cap.read()
        if ret:
            try:
                img = detect(img)
            except Exception as e:
                print('pizd', e)
                continue
            success, encoded_img = cv2.imencode('.jpeg', img)
            if success:
                base64Img = 'data:image/jpeg;base64,'+base64.b64encode(encoded_img).decode()
                person = {
                    'name': 'dima',
                    'date': t.time()
                }
                await websocket.send(json.dumps({'image': base64Img, 'person': person}))
                print(f"Elapsed {t.time()-start_time}")

start_server = websockets.serve(time, '127.0.0.1', 5678)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

cap.release()

cv2.destroyAllWindows()