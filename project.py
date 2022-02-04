import numpy as np
import cv2
from matplotlib import pyplot as plt

pts_move = []
pts1 = []
pts2 = [[0, 0], [300, 0], [300, 300], [0, 300]]


def onClick(event, x, y, flags, param):
    global pts_move
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts1) < 4:
            pts1.append([x, y])
            print(pts1)
    if event == cv2.EVENT_MOUSEMOVE:
        if len(pts1) >= 1:
            pts_move = [x, y]


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', onClick)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    _, frame = cap.read()
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(pts1) >= 2:
        for i in range(0, len(pts1) - 1):
            cv2.line(frame, pts1[i], pts1[i + 1], (0, 255, 0))

    if len(pts1) < 4 and pts_move != []:
        cv2.line(frame, pts1[-1], pts_move, (0, 255, 0))
        if len(pts1) == 3:
            cv2.line(frame, pts1[0], pts_move, (0, 255, 0))

    if len(pts1) == 4:
        T = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
        frame = cv2.warpPerspective(imgray, T, (300, 300))

        for row in range(3):
            for col in range(3):
                start_point = [15+(col*100), 15+(row*100)]
                end_point = [85+(col*100), 85+(row*100)]
                cv2.rectangle(frame, start_point, end_point, (255, 0, 0))

    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # thresh = cv2.medianBlur(thresh, 5)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours) != 0:
    #     # draw in blue the contours that were founded
    #     cv2.drawContours(frame, contours, -1, 255, 3)
    #     # find the biggest countour (c) by the area
    #     c = max(contours, key=cv2.contourArea)
    #
    #     approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
    #     cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)
    #
    #     if len(approx) == 4:
    #         position = [i[0] for i in approx]  #position for perspective

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
