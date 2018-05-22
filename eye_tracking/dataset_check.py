import cv2
import numpy as np
import os
import time
def imgshow(i, box_info) :
    name = 'dataset/%d.jpg'%(i)




    cv2.namedWindow(name)
    img = cv2.imread('dataset/%d.jpg'%(i))
    center_x, center_y, width, height = box_info[0][0], box_info[0][1], box_info[0][2], box_info[0][3]
    pt1_x, pt1_y, pt2_x, pt2_y = center_x - (width / 2), center_y + (height / 2), center_x + (width / 2), center_y - (width / 2)
    pt1 = (pt1_x, pt1_y)
    pt2 = (pt2_x, pt2_y)
    while True :
        cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(225, 0, 0), thickness=1)
        cv2.imshow(name, img)

        if cv2.waitKey(0) :
            cv2.destroyAllWindows()
            break

if __name__=="__main__" :

    for i in range(1000) :

        list = np.loadtxt('dataset/eyesPos.csv', dtype='int', delimiter=',')
        box_info = list[i][1:]
        imgshow(i, box_info)