import tensorflow as tf
import numpy as np
import cv2
import os

img_width = 640
img_height = 480

os.chdir('/home/leejin/git/image_processing/eye-tracking')



def next_batch(i, batch_size):
    os.chdir('/home/leejin/git/image_processing/eye-tracking')
    list = np.loadtxt('dataset/3/eyesPos.csv', dtype='int', delimiter=',')

    start = i * batch_size
    end = start + batch_size

    img_list = cv2.imread('dataset/3/%d.jpg' % start)

    list = list[start:end+1][1:]
    label = np.array([0, 0, 0, 0])
    img_list = img_list[np.newaxis, :, :, :]
    for j in range(start + 1, end):
        new_img = cv2.imread('dataset/3/%d.jpg' % j)

        new_img = new_img[np.newaxis, :, :, :]
        img_list = np.concatenate((img_list, new_img))

    for k in range(batch_size) :
        list_tmp = (list[k][0], list[k][1], list[k][2], list[k][3])
        label = np.vstack((label, list_tmp))

    label = label[1:][:]
    return (img_list, label)




img = cv2.imread('dataset/0/0.jpg')

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
L4 = tf.reshape(L4, [-1, 30*40*256])

W5 = tf.get_variable('W5', [30*40*256, 2048], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L5 = tf.matmul(L4, W5)
L5 = tf.nn.relu(L5)

W6 = tf.get_variable('W6', [2048, 512], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L6 = tf.matmul(L5, W6)
L6 = tf.nn.relu(L6)

W7 = tf.get_variable('W7', [512, 64], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
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




cost = tf.reduce_mean(tf.square((L10 - Y)))
optimizer = tf.train.AdadeltaOptimizer(0.05).minimize(cost)


ckpt = tf.train.get_checkpoint_state('./model_save_big')

init = tf.global_variables_initializer()
sess = tf.Session()

saver = tf.train.Saver(tf.global_variables())

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) :
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("reload")
else :
    sess.run(init)


batch_size = 2
total_batch = 22

for epoch in range(10) :
    total_cost = 0

    for i in range(total_batch) :
        batch_xs, batch_ys = next_batch(i, batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y:batch_ys})

        total_cost += cost_val

    print('Epoch : ', '%04d'%(epoch+1), 'avg. cost =','{:.4f}'.format(total_cost/total_batch))

    save_path = saver.save(sess, "model_save_big/eye_track_model.ckpt")
    print("Model saved in path: %s" % save_path)


