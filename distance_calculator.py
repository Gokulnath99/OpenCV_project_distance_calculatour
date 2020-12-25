import math
import cv2
import numpy as np
import threading

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty():
    pass


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 29, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 40, 255, empty)
cv2.createTrackbar("Area", "Parameters", 15000, 30000, empty)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


g = True
d = 0
k = 0
p = []
s = []


def getDistance(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def change():
        threading.Timer(5.0, change).start()
        global g
        g = not g

    for cnt in contours:
        global g
        global p
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        def printit():
            p.append(cx)
            p.append(cy)
            if len(p) == 4:
                global d
                d = np.sqrt(((p[0] - p[2]) ** 2) + ((p[1] - p[3]) ** 2))
                speed = d/5
                angle = math.atan2(p[1] - p[3], p[0] - p[2]) * 180/math.pi
                p.clear()
                print(d)
                f1 = open("distance.txt", "a")
                f1.write("Distance traveled in 5sec : {} \n".format(d))
                f1.close()

                f2 = open("speed.txt", "a")
                f2.write("speed : {} \n".format(speed))
                f2.close()

                f3 = open("angle.txt", "a")
                f3.write("Angle : {} \n".format(angle))
                f3.close()
                s.append(speed)
                if len(s) == 2:
                    acceleration = (s[1] - s[0]) / 10
                    f4 = open("acceleration.txt", "a")
                    f4.write("acceleration : {} \n".format(acceleration))
                    f4.close()
                    s.clear()

                cv2.putText(imgContour, "{}".format(int(d)), (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 0), 3)
            cv2.circle(imgContour, (cx, cy), 7, (255, 255, 255), -1)

            cv2.putText(imgContour, "Center", (cx - 20, cy - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(imgContour, "x: " + str(cx), (cx - 20, cy - 35), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(imgContour, "y: " + str(cy), (cx - 20, cy - 50), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 255, 0), 1)

        if g:
            global k
            printit()
            if k == 0:
                change()
                k = 1 + k
            else:
                g = False
                k = 1 + k
        else:
            continue


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")

        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)


while True:
    success, img = cap.read()
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDil, imgContour)
    getDistance(imgDil, imgContour)
    imgStack = stackImages(0.6, ([img, imgCanny],
                                 [imgDil, imgContour]))
    cv2.imshow("Result", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
