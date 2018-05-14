import cv2
import numpy as np
import os
import time
def imgshow(i, num, box_info) :

    cv2.namedWindow('test')
    img = cv2.imread('dataset/%d/%d.jpg'%(num,i))
    pt1 = (box_info[0], box_info[1])
    pt2 = (box_info[2], box_info[3])
    while True :
        cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(225, 0, 0), thickness=1)
        cv2.imshow('test', img)

        if cv2.waitKey(0) :
            cv2.destroyAllWindows()
            break
if __name__=="__main__" :
    num = input('number of folder')
    num = int(num)
    for i in range(1000) :

        os.chdir('/home/leejin/git/image_processing/eye-tracking')
        list = np.loadtxt('dataset/%d/eyesPos.csv'%num, dtype='int', delimiter=',')
        box_info = list[i][1:]
        imgshow(i, num, box_info)
