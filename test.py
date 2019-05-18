import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("eye_recording.flv")

while True:
    ret, frame = cap.read()
    if ret is False:
        break
    roi = frame
    #roi = frame[269: 795, 537: 1416]
    rows = roi.shape[0]
    cols = roi.shape[1]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    ################ ОГРАНИЧИТЕЛЬ ПО ПЛОЩАДИ КОНТУРОВ
    # Создадим массив для хранения размеров площадей внутри контуров
    squares = [i for i in range(len(contours))]
    # Создаем пустой массив для хранения нужных контуров
    contoursTrue = []
    for i in range(len(contours)):
        squares[i] = cv2.contourArea(contours[i])
        if (squares[i] > 50) and (squares[i] < 1000):
            contoursTrue.append(contours[i])
    ##################################################

    contours = sorted(contoursTrue, key=(lambda x: cv2.contourArea(x)), reverse=True)
    frame = cv2.drawContours(roi, contoursTrue, -1, (0, 0, 255), 3)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_roi)
    cv2.imshow("Roi", roi)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()