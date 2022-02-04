import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


while True:
    _, frame = cap.read()
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    thresh = cv2.medianBlur(thresh, 5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(frame, contours, -1, 255, 3)
        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)

        approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
        cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)

        if len(approx) == 4:
            position = [i[0] for i in approx]  #position for perspective



    cv2.imshow('t', frame)
    cv2.waitKey(1)