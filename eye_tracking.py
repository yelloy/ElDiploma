import numpy as np
import cv2

def find_centre_mass(countours, image):
    kpCnt = len(contours[0])

    x = 0
    y = 0

    for kp in contours[0]:
        x = x + kp[0][0]
        y = y + kp[0][1]

    #cv2.circle(image, (np.uint8(np.ceil(x / kpCnt)), np.uint8(np.ceil(y / kpCnt))), 1, (0, 0, 255), 3)
    return (np.uint8(np.ceil(x / kpCnt)), np.uint8(np.ceil(y / kpCnt)))

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("output.avi")

# Захват видео для сохранения
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# video recorder
#fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
#video_writer = cv2.VideoWriter("output.avi", fourcc, 20, (w, h))

first_cycle_skipped = 0
difference_x = 0
difference_y = 0
difference_first_x = 0
while 1:
    ret, img = cap.read()
#    video_writer.write(img)

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
                cv2.imshow('gray puple', gray_roi)
                _, light_area = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)

                gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                _, threshold = cv2.threshold(gray_roi, 30, 255, cv2.THRESH_BINARY_INV)

                light_area = cv2.bitwise_not(light_area)

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

                cv2.drawContours(eye_image, contoursTrue, 0, (0, 0, 255), 1)
                cnt_mass_center_y = []  # массив для хранения центров масс
                cnt_mass_center_x = []  # массив для хранения центров масс

                coord_x, coord_y = find_centre_mass(contoursTrue, image=eye_image)

                '''
                for cnt in contoursTrue:
                    # Находим момент контура
                    moment = cv2.moments(cnt)
                    # Отсеиваем вероятное деление на ноль. Не знаю почему, но есть нулевой момент равный нулю
                    if moment['m00'] == 0:
                        pass
                    # Считаем центр масс каждого контура
                    else:
                        coord_x = int(moment['m01'] / moment['m00'])
                        coord_y = int(moment['m10'] / moment['m00'])
                        cnt_mass_center_y.append(coord_y)
                        cnt_mass_center_x.append(coord_x)'''

                #print(cnt_mass_center_x, cnt_mass_center_y)
                # print(first_cycle_skipped)

                # Глаз всегда расположен ниже брови, поэтому находим его по расположению центра масс
                '''if cnt_mass_center_x[0] > cnt_mass_center_x[1]:
                    eye_contour = contoursTrue[0]
                    coord_x = cnt_mass_center_x[0]
                    coord_y = cnt_mass_center_y[0]
                else:
                    eye_contour = contoursTrue[1]
                    coord_x = cnt_mass_center_x[1]
                    coord_y = cnt_mass_center_y[1]'''
                # print("ARBUZ")
                if first_cycle_skipped:
                    difference_x = ex + ew//2 - coord_x
                    #difference_y = ey + eh//2 - coord_y

                    difference_x_final = int(difference_x - difference_first_x)
                    print(difference_x_final)

                    threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)

                    #cv2.line(threshold, (cnt_mass_center_x[0], cnt_mass_center_y[0] - 1), (cnt_mass_center_x[0], cnt_mass_center_y[0] + 1), (0, 255, 0), 2)
                    #cv2.line(threshold, (cnt_mass_center_x[1], cnt_mass_center_y[1] - 1), (cnt_mass_center_x[1], cnt_mass_center_y[1] + 1), (255, 255, 0), 2)

                    cv2.line(threshold, (coord_x, coord_y - 1), (coord_x, coord_y + 1), (0, 255, 0), 2)
                    cv2.line(eye_image, (coord_x, coord_y - 1), (coord_x, coord_y + 1), (0, 255, 0), 2)
                    cv2.imshow('threshold', threshold)
                    cv2.imshow("eye_eee", eye_image)

                    # print("eye_pixel_value", eye_image[eye_image.shape[0]//2][5])
                else:
                    difference_first_x = ex + ew//2 - coord_x - 3

                first_cycle_skipped = 1
            #img = cv2.resize(gray_roi, (960, 540))

        except:
            print("No eyes")
    except:
        print("No faces")
            #cv2.imshow('eye_img', eye_image)


    cv2.rectangle(img, (5, img.shape[0] - 35), (img.shape[1]-5, img.shape[0]-5), (255, 255, 255), 5)
    try:
        if difference_x_final > 4:
            difference_x_final = 5
        else:
            if difference_x_final < -4:
                difference_x_final = -5
            else:
                difference_x_final = 0

        cv2.line(img, (img.shape[1]//2 - difference_x_final*5, img.shape[0]-35), (img.shape[1]//2 - difference_x_final*5, img.shape[0]-5), (255, 255, 255), 4)
    except:
        pass
    cv2.line(img, (img.shape[1]//2, img.shape[0]-35), (img.shape[1]//2, img.shape[0]-5), (0, 0, 255), 2)
    cv2.imwrite("eye_tracker_result_full_image.png", img)

    #img = cv2.resize(img, (320,180))

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#video_writer.release()
cap.release()
cv2.destroyAllWindows()
