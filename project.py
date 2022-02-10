import numpy as np
import cv2
from matplotlib import pyplot as plt
from TicTaeToe import TicTaeToe
import time

xo = TicTaeToe()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
bgsub = cv2.createBackgroundSubtractorKNN()


AR = cv2.imread('2-platform.png', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
kp1, des1 = sift.detectAndCompute(AR, None)

while True:
    _, frame = cap.read()
    # frame = cv2.flip(frame, -1)
    # frame = img
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(imgray, None)
    matched = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matched:
        if m.distance < 0.6 * n.distance:
            good.append(m)
    if len(good) >= 4:
        # M = cv2.drawMatches(AR, kp1, img2, kp2, good, None, flags=2)
        # cv2.imshow('M', M)
        pt1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pt2 = np.float32([kp2[m.trainIdx].pt for m in good])
        T1, _ = cv2.findHomography(pt2, pt1, cv2.RANSAC, 5.0)
        if T1 is not None:
            print(len(T1))
            frame = cv2.warpPerspective(imgray, T1, AR.shape)

            frame = frame[50:-50,50:-50]
            frame = cv2.resize(frame,(300,300))

            fg = bgsub.apply(frame)
            B_fg = cv2.threshold(fg, 20, 255, cv2.THRESH_BINARY)[1]

            B = cv2.threshold(frame, 60, 255, cv2.THRESH_BINARY)[1]
            B = cv2.bitwise_not(B)
            B = cv2.erode(B, np.ones((2, 2)))

            cv2.imshow('frame', frame)
            cv2.imshow('b', B)
            cv2.imshow('gray', B_fg)

            for row in range(3):
                for col in range(3):
                    start_point = [15 + (col * 100), 15 + (row * 100)]
                    end_point = [85 + (col * 100), 85 + (row * 100)]
                    cv2.rectangle(frame, start_point, end_point, (0, 0, 0))

                    area = B[start_point[1]:start_point[1] + 70, start_point[0]:start_point[0] + 70]

                    if (np.sum(area == 255) / (70 * 70) > 0.1) and np.sum(B_fg == 255) < 1500:
                        if xo.board[row, col] == '_':
                            print('....Waiting ai....')
                            index = xo.move_vs_ai((row, col))
                            result = xo.display()
                            if result == 'win':
                                print("########################")
                                print("####  You Win !!!  ####")
                                print("########################")
                                break
                            elif result == 'lose':
                                print("########################")
                                print("####  You lose !!!  ####")
                                print("########################")
                                break
                            elif result == 'draw':
                                print("########################")
                                print("####  You Draw !!!  ####")
                                print("########################")
                                break
                    if xo.board[row, col] == 'O':
                        org = [15 + (col * 100), 85 + (row * 100)]
                        cv2.putText(frame, 'O', org, cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 3)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    time.sleep(0.1)
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


