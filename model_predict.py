import tensorflow as tf
import numpy as np
import cv2
import os

img_width = 100
img_height = 100

os.chdir('/home/leejin/git/image_processing/eye-tracking')
2

X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
Y = tf.placeholder(tf.float32, [None, 4])

W1 = tf.get_variable('W1',[3, 3, 3, 32],dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L1 = tf.nn.conv2d(X, W1,strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W2 = tf.get_variable('W2',[3, 3, 32, 64],dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L2 = tf.nn.conv2d(L1, W2,strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W3 = tf.get_variable('W3',[3, 3, 64, 128],dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L3 = tf.nn.conv2d(L2, W3,strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W4 = tf.get_variable('W4',[3, 3, 128, 256],dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L4 = tf.nn.conv2d(L3, W4,strides=[1,1,1,1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L4 = tf.reshape(L4, [-1, 7*7*256])

W5 = tf.get_variable('W5', [7*7*256, 512], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L5 = tf.matmul(L4, W5)
L5 = tf.nn.relu(L5)

W6 = tf.get_variable('W6', [512, 256], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L6 = tf.matmul(L5, W6)
L6 = tf.nn.relu(L6)

W7 = tf.get_variable('W7', [256, 64], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L7 = tf.matmul(L6, W7)
L7 = tf.nn.relu(L7)

W8 = tf.get_variable('W8', [64, 8], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L8 = tf.matmul(L7, W8)
L8 = tf.nn.relu(L8)

W9 = tf.get_variable('W9', [8, 8], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L9 = tf.matmul(L8, W9)
L9 = tf.nn.relu(L9)

W10 = tf.get_variable('W10', [8, 4], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())

L10 = tf.matmul(L9, W10)


ckpt = tf.train.get_checkpoint_state('./model_save_big')

init = tf.global_variables_initializer()
sess = tf.Session()

saver = tf.train.Saver(tf.global_variables())


saver.restore(sess, ckpt.model_checkpoint_path)
print("reload")


def imgshow(i, num) :

    cv2.namedWindow('test')
    img = cv2.imread('dataset/%d/%d.jpg'%(num,i))
    label = list[i][:]

    img2 = cv2.resize(img, (100, 100))
    img2 = img2[np.newaxis,:,:,:]
    tmp = sess.run(L10, feed_dict={X:img2})
    pt1_x, pt1_y, pt2_x, pt2_y = tmp[0][0], tmp[0][1], tmp[0][2], tmp[0][3]
    pt1 = (pt1_x, pt1_y)
    pt2 = (pt2_x, pt2_y)
    print(pt1, pt2)
    print((label[1], label[2]), (label[3], label[4]))
    while True :

        cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(225, 0, 0), thickness=3)
        cv2.rectangle(img, pt1=(label[1], label[2]), pt2=(label[3], label[4]), color=(0,225,0), thickness=3 )
        cv2.imshow('test', img)

        if cv2.waitKey(0) :
            cv2.destroyAllWindows()
            break

num = input('number of folder')
num = int(num)
for i in range(1000) :

    list = np.loadtxt('dataset/%d/eyesPos.csv'%num, dtype='int', delimiter=',')

    imgshow(i, num)


