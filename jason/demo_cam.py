#! /usr/bin/env python

import cv2

cv2.namedWindow('preview')
cap = cv2.VideoCapture(0)
width = cap.get(3)
height = cap.get(4)
cap.set(3, width / 2.0)
cap.set(4, height / 2.0)
rval, frame = cap.read()

while True:
    if frame is not None:
        cv2.imshow('preview', frame)
        print frame.mean()
    rval, frame = cap.read()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
