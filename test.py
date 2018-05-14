import numpy as np
import cv2
import os

list = np.loadtxt('dataset/3/eyesPos.csv', dtype='int', delimiter=',')
print(list.shape[0])