import cv2
import numpy as np
import os, sys

drawing = False
drawn = False
img = None
img2 = None
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

def draw_rectangle(last_num) :
    global center_x, center_y, width, height, drawn, list, list2, img, img2
    if last_num :
        j = last_num
        print("j : %d"%j)
    else :
        j = 0
    list = []
    list2 = []
    while True :
        name = 'dataset/%d.jpg'%j

        img = cv2.imread(name)

        print('name : ', name)
        cv2.namedWindow('test')
        cv2.moveWindow('test', 800, 0)
        cv2.setMouseCallback('test', draw_rect, param=img)
        while True :
            try:
                cv2.imshow('test', img)
            except :
                print('no image')
                final_num = str(j)
                fo.write(final_num)
                # np.savetxt(fo, final_num, fmt="%d")
                sys.exit()

                print('no image')
                final_num = str(j)
                fo.write(final_num)
                # np.savetxt(fo, final_num, fmt="%d")
                sys.exit()

            img2 = img.copy()
            k = cv2.waitKey(0)
            if k:
                if k & 0xFF == ord('q'):
                    print('brak')
                    final_num = str(j)
                    fo.write(final_num)
                    #np.savetxt(fo, final_num, fmt="%d")
                    break
                else :
                    break
        if drawn:
            """
            pt1 = (int(center_x - (width / 2)), int(center_y + (height / 2)))
            pt2 = (int(center_x + (width / 2)), int(center_y - (height / 2)))
            list = np.array([[j, pt1[0], pt1[1], pt2[0], pt2[1]]])
            """
            list = ([center_x, center_y, width, height])
            print('list : ', list)
            np.savetxt(f, list, fmt='%d', delimiter=',')
        j += 1

        cv2.destroyAllWindows()

if __name__ == "__main__" :
    last_num = None
    try:
        fo = open('dataset/last_num.txt', 'r')
        print('mode r')
    except:
        fo = open('dataset/last_num.txt', 'w')
        print('mode w')

    if fo.mode == 'r':
        num = fo.readline()
        fo = open('dataset/last_num.txt', 'w')
        if num=='':
            last_num = 0
            print("last_num : %d" % last_num)
        else:

            last_num = int(num)
            print("last_num : %d" % last_num)
    else:
        last_num = 0
    f = open('dataset/eyesPos.csv', 'a+')
    draw_rectangle(last_num)