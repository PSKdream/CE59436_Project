import numpy as np
import cv2
from matplotlib import pyplot as plt
from TicTaeToe import TicTaeToe

xo = TicTaeToe()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

AR = cv2.imread('2-platform.png', cv2.IMREAD_GRAYSCALE)
AR2 = cv2.imread('2New-platform.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
kp1, des1 = sift.detectAndCompute(AR, None)

kp3, des3 = sift.detectAndCompute(AR2, None)

iframe = 0
n_frame = None
status = None
result = '?'

while True:
    _, frame = cap.read()

    cv2.imshow('original',cv2.flip(frame, -1))

    # frame = img
    iframe += 1
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
            #Check
            matched = matcher.knnMatch(des3, des2, k=2)
            good = []
            for m, n in matched:
                if m.distance < 0.6 * n.distance:
                    good.append(m)
            M = cv2.drawMatches(AR2, kp3, imgray, kp2, good, None, flags=2)
            cv2.imshow('M', M)

            # if len(good) < 1 :
            #     print("Not Confirm")
            # # print(len(T1))
            frame = cv2.warpPerspective(frame, T1, AR.shape)

            frame = frame[50:-50,50:-50]
            frame = cv2.resize(frame,(300,300))


            B = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)[1]
            B = cv2.bitwise_not(B)
            B = cv2.erode(B, np.ones((2, 2)))

            cv2.imshow('frame', frame)
            cv2.imshow('B1111', B)

            if status is None and len(good) < 1:
                status = "Closed"
            elif status == "Closed" and len(good) >= 2:
                status = "Opened"

            for row in range(3):
                for col in range(3):
                    start_point = [15 + (col * 100), 15 + (row * 100)]
                    end_point = [85 + (col * 100), 85 + (row * 100)]
                    # cv2.rectangle(frame, start_point, end_point, (0, 0, 0))
                    area = B[start_point[1]:start_point[1] + 70, start_point[0]:start_point[0] + 70]

                    if xo.board[row, col] == 'O':
                        org = [15 + (col * 100), 85 + (row * 100)]
                        cv2.putText(frame, 'O', org, cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 3)

                    if result != '?':
                        cv2.putText(frame, result ,(100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),2)
                        continue

                    if status != "Opened":
                        continue

                    if n_frame is None:
                        n_frame = iframe
                    if iframe == n_frame + 15:
                        # print("alert")
                        if (np.sum(area == 255) / (70 * 70) > 0.08):
                            if xo.board[row, col] == '_':
                                if status == "Opened":
                                    n_frame = None
                                    status = None
                                result = xo.validateResult()
                                if result == '?':
                                    print('....Waiting ai....')
                                    index = xo.move_vs_ai((row, col))
                                    result = xo.validateResult()
                                    print(xo.board)
                                    if result == '?':
                                        print("----Your turn----")
                                    else:
                                        print("----GameOver----")

                            elif xo.board[row, col] == 'O':
                                print('Please try again')
                                if status == "Opened":
                                    n_frame = None
                                    status = None


                                # if result == 'win':
                                #     print("########################")
                                #     print("####  You Win !!!  ####")
                                #     print("########################")
                                #     break
                                # elif result == 'lose':
                                #     print("########################")
                                #     print("####  You lose !!!  ####")
                                #     print("########################")
                                #     break
                                # elif result == 'draw':
                                #     print("########################")
                                #     print("####  You Draw !!!  ####")
                                #     print("########################")
                                #     break




    cv2.imshow('frame', frame)
    cv2.waitKey(50)


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

