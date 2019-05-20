import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

first_cycle_skipped = 0
difference_x = 0
difference_y = 0

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        x,y,w,h = faces[0]
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_image = img[y:(y + h), x:(x + w)]
        roi_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        sortlist = (eyes[0][0], eyes[1][0])
        if sortlist[0] > sortlist [1]:
            eye = eyes[0]
        else:
            eye = eyes[1]
        #print("eyes", sortlist)
        try:
            ex, ey, ew, eh = eye
            #cv2.rectangle(face_image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            if ew > 30 and eh > 30:
                eye_image = img[ey + y:(ey + y + eh), ex + x:(ex + x + ew)]

                rows = eye_image.shape[0]
                cols = eye_image.shape[1]

                #### Обработаем изображение по цвету.

                gray_roi = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

                _, threshold = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY_INV)

                cv2.imshow('threshold', threshold)

                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                ################ ОГРАНИЧИТЕЛЬ ПО ПЛОЩАДИ КОНТУРОВ внутри глаза
                # Создадим массив для хранения размеров площадей внутри контуров
                squares = [i for i in range(len(contours))]
                # Создаем пустой массив для хранения нужных контуров
                contoursTrue = []
                for i in range(len(contours)):
                    squares[i] = cv2.contourArea(contours[i])
                    if (squares[i] > 10) and (squares[i] < 1000):
                        contoursTrue.append(contours[i])
                        ##################################################

                contours = sorted(contoursTrue, key=(lambda x: cv2.contourArea(x)), reverse=True)
                #eye_image = cv2.drawContours(eye_image, contoursTrue, -1, (0, 0, 255), 3)

                for cnt in contoursTrue:
                    # Находим момент контура
                    moment = cv2.moments(cnt)
                    # Отсеиваем вероятное деление на ноль. Не знаю почему, но есть нулевой момент равный нулю
                    if moment['m00'] == 0:
                        pass
                    # Считаем центр масс каждого контура и рисуем на нем прицел
                    else:
                        coord_x = int(moment['m01'] / moment['m00'])
                        coord_y = int(moment['m10'] / moment['m00'])

                        if first_cycle_skipped:
                            difference_x = old_x - coord_x
                            difference_y = old_y - coord_y

                        old_x = coord_x
                        old_y = coord_y

                        first_cycle_skipped = 1

                        cv2.line(eye_image, (coord_x, coord_y - 2), (coord_x, coord_y + 2), (0, 255, 0), 4)
                        cv2.imwrite("eye_tracker_result.png", img)
                    break

        except:
            print("No eyes")
    except:
        print("No faces")
            #cv2.imshow('eye_img', eye_image)

    cv2.rectangle(img, (5, img.shape[0] - 35), (img.shape[1]-5, img.shape[0]-5), (255, 255, 255), 5)
    cv2.line(img, (img.shape[1]//2 + difference_x*5, img.shape[0]-35), (img.shape[1]//2 + difference_x*5, img.shape[1]-5), (255, 255, 255), 4)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()