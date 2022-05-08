#Import the neccesary libraries
import numpy as np
import argparse
import cv2 
# from error import servo_error
import serial 
import serial.tools.list_ports
import time

# Get model weights and configuration paths
MODEL_WEIGHTS = "yolov4\\yolov4-custom_best.weights"
CONFIG_PATH = "yolov4\\yolov4-custom.cfg"

cap = cv2.VideoCapture(1)

classes = ['white_boat']

net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, MODEL_WEIGHTS)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416,416))

# ServoController = serial.Serial("COM14", 115200, timeout=0.1)
# time.sleep(3)

# ServoController.write(str.encode('0, 0, 0, 0, 0'))
while True:
    # ret, frame = cap.read()
    # box = [frame.shape[1]/2, frame.shape[0]/2, 200, 300]
    frame = cv2.imread("Image350.jpg")
    # frame = cv2.resize(frame, [640, 480])
    classIds, scores, boxes = model.detect(frame, confThreshold=0.3, nmsThreshold=0.2)

    for (classId, score, box) in zip(classIds, scores, boxes):
        top_left = (box[1], box[0])
        bottom_right = (box[1] + box[3], box[0] + box[2])

        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0,0,255), thickness = 1)
        # print(classes[classId])        
        text = "{0}: {1}".format(classes[classId], str(round(score,2)))
        cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.33, color=(0,0,255), thickness=1)

    # Display the image with predictions
    cv2.imshow('window', frame)
    cv2.waitKey(1) # milliseconds
    # print(len(boxes))
    if len(boxes) == 0:
        box_empty = 1
    else:
        box_empty = 0

    bb_center_x = int(box[1] + box[3]/2)
    bb_center_y = int(box[0] + box[2]/2)
    # print(f'Bb_center_x: {bb_center_x}')
    frame_center_x = int(frame.shape[1]/2)
    frame_center_y = int(frame.shape[0]/2)
    # sum = bb_center_x + bb_center_y + frame_center_x + frame_center_y


    # ServoController.write(str.encode(str(bb_center_x) + ',' + str(bb_center_y) + ',' + str(frame_center_x) + ',' + str(frame_center_y) + ',' + str(box_empty) + ','))
    # ServoController.write(str.encode(str(2) + ',' + str(4) + ',' + str(5) + ',' + str(5) + ','))
    # message = ServoController.read_until()
    # print(message)

