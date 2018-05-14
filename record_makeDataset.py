import numpy as np
import cv2
import sys
import os

recordStart = False
recordEnd = False
drawing = False
drawn = False
img = None
img2 = None
center_x, center_y, width, height = None, None, None, None
def onMouse(event, x, y, flags, param) :
    global recordStart, recordEnd
    if event == cv2.EVENT_LBUTTONDOWN :
        print('record start')
        recordStart = True

    elif event == cv2.EVENT_LBUTTONUP :

        recordStart = False
        recordEnd = True

def draw_rect(event, x, y, flags, param) :
    global drawing, ix, iy, center_x, center_y, width, height, drawn, img, img2
    if event == cv2.EVENT_LBUTTONDOWN :
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE :
        if drawing :
            cv2.rectangle(img, (ix, iy), (x, y), (0,255,0), 1)
            cv2.imshow('test', img)
            img = img2.copy()

    elif event == cv2.EVENT_LBUTTONUP :
        drawing = False
        drawn = True
        cv2.rectangle(param, (ix, iy), (x, y), (0, 255, 0), 1)
        center_x, center_y = int((x + ix)/2), int((y + iy)/2)
        width, height = int((x - ix)), int((y - iy))


def writeVideo(i) :

    global recordStart, recordEnd
    try :
        print('camera start')
        cap = cv2.VideoCapture(0)


    except :
        print('camera setting failed')


    j = 0
    while True :
        ret, frame = cap.read()

        if not ret :
            print('video reading error')
            break

        cv2.imshow('video', frame)
        cv2.setMouseCallback('video', onMouse)
        if recordStart:
            cv2.imwrite('dataset/' + '%d'%i + '/%d.jpg'%j, frame)
            j += 1
        if cv2.waitKey(1) & 0xFF == ord('q') :
            sys.exit()
        elif recordEnd == True :
            print('finish record')
            recordEnd = False
            break


    cap.release()


    cv2.destroyAllWindows()


def draw_rectangle(i) :
    global center_x, center_y, width, height, drawn, list, list2, img, img2
    j = 0
    list = np.array([0,0,0,0,0])
    list2 = np.array([0,0,0,0,0])
    while True :
        name = 'dataset/%d/%d.jpg'%(i, j)

        img = cv2.imread(name)

        print('name : ', name)
        cv2.namedWindow('test')
        cv2.moveWindow('test', -10, -10)
        cv2.setMouseCallback('test', draw_rect, param=img)
        while True :
            try:
                cv2.imshow('test', img)
            except :

                print('no image')
                sys.exit()
            img2 = img.copy()
            if cv2.waitKey(0) :
                break
        if drawn:
            pt1 = (int(center_x - (width / 2)), int(center_y + (height / 2)))
            pt2 = (int(center_x + (width / 2)), int(center_y - (height / 2)))
            list2 = (j, pt1[0], pt1[1], pt2[0], pt2[1])
            list = np.vstack((list, list2))
            print('list : ', list)
            np.savetxt('dataset/%d/eyesPos.csv' % i, list, fmt='%d', delimiter=',', newline='\n', footer='num, pt1_x, pt1_y, pt2_x, pt2_y')

        j += 1

        cv2.destroyAllWindows()
if __name__ == "__main__" :
    start = input('start point')
    start = int(start)
    for i in range(start, 10000) :
        os.chdir('/home/leejin/git/image_processing/eye-tracking')

        try :
            os.mkdir('dataset/%d'%i, 0o777)
        except :
            pass

        writeVideo(i)
        will = input('do you want to draw rectangles right now? yes : y, no : n')
        if (will == 'y') :
            draw_rectangle(i)
        else :
            break