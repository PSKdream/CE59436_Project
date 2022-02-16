import numpy as np
import cv2
from matplotlib import pyplot as plt
from TicTaeToe import TicTaeToe

# Init SIFT and Matcher
sift = cv2.SIFT_create()

# Read ARTable Marker
ARTable = cv2.imread('ARMarker_Table.png', cv2.IMREAD_GRAYSCALE)
ARConfirm = cv2.imread('ARMarker_Confirm.png', cv2.IMREAD_GRAYSCALE)
kpTable, desTable = sift.detectAndCompute(ARTable, None)
kpConfirm, desConfirm = sift.detectAndCompute(ARConfirm, None)

# Init TicTaeToe
xo = TicTaeToe()

# Init Variable
iframe = 0
n_frame = None
state = None
result = '?'
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def knnMatch(des_ar_marker, des_image):
    tempArr = []
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matched = matcher.knnMatch(des_ar_marker, des_image, k=2)
    for m, n in matched:
        if m.distance < 0.6 * n.distance:
            tempArr.append(m)
    return tempArr
# return distance

while True:
    _, frame = cap.read()  # read camera
    cv2.imshow('original', cv2.flip(frame,-1))  # show original image
    iframe += 1

    # detect AR Marker table
    kpImage, desImage = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    goodPoint = knnMatch(desTable, desImage)

    if len(goodPoint) >= 4:

        pt1 = np.float32([kpTable[m.queryIdx].pt for m in goodPoint])
        pt2 = np.float32([kpImage[m.trainIdx].pt for m in goodPoint])
        T1, _ = cv2.findHomography(pt2, pt1, cv2.RANSAC, 5.0)
        if T1 is not None:
            # detect AR Marker confirm
            goodPoint = knnMatch(desConfirm, desImage)

            # Perspective And Crop
            frame = cv2.warpPerspective(frame, T1, ARTable.shape)
            frame = frame[50:-50, 50:-50]
            frame = cv2.resize(frame, (300, 300))

            # Pre-Process, threshold to BIN
            B = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)[1]
            B = cv2.bitwise_not(B)
            B = cv2.erode(B, np.ones((2, 2)))

            # Show Image
            cv2.imshow('frame', frame)
            cv2.imshow('Bin', B)

            # change state confirm
            if state is None and len(goodPoint) < 1:
                state = "Closed"
            elif state == "Closed" and len(goodPoint) >= 2:
                state = "Opened"

            # check mark in the paper
            for row in range(3):
                for col in range(3):
                    # draw O
                    if xo.board[row, col] == 'O':
                        org = [15 + (col * 100), 85 + (row * 100)]
                        cv2.putText(frame, 'O', org, cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 3)

                    # check game result
                    if result != '?':
                        cv2.putText(frame, result, (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                        continue

                    # check confirm
                    if state != "Opened":
                        continue

                    if n_frame is None:
                        n_frame = iframe
                    elif iframe == n_frame + 15:
                        # Init position and area
                        start_point = [15 + (col * 100), 15 + (row * 100)]
                        end_point = [85 + (col * 100), 85 + (row * 100)]
                        area = B[start_point[1]:start_point[1] + 70, start_point[0]:start_point[0] + 70]

                        # check player mark X
                        if np.sum(area == 255) / (70 * 70) > 0.08:
                            if xo.board[row, col] == '_':
                                n_frame = None
                                state = None
                                if result == '?':
                                    print('....Waiting ai....')
                                    result = xo.play((row, col))  # Play and get game result
                                    print(xo.board)
                                    if result == '?':
                                        print("----Your turn----")
                                    else:
                                        print("----GameOver----")

                            elif xo.board[row, col] == 'O':
                                print('Please try again')
                                if state == "Opened":
                                    n_frame = None
                                    state = None
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
