import tensorflow as tf
import numpy as np
import cv2
import os

img_width = 100
img_height = 100



def next_batch(i, batch_size):
    list = np.loadtxt('dataset/eyesPos.csv', dtype='int', delimiter=',')

    start = i * batch_size
    end = start + batch_size

    img_list = cv2.imread('dataset/%d.jpg' % start, cv2.IMREAD_GRAYSCALE)
    img_list = cv2.resize(img_list, (100,100))
    img_list = np.resize(img_list, (100,100,1))
    list = list[start:end+1][1:]
    #print('list : ', list)
    label = np.array([0, 0, 0, 0])
    img_list = img_list[np.newaxis, :, :, :]
    for j in range(start + 1, end):
        new_img = cv2.imread('dataset/%d.jpg' % j)
        new_img = cv2.resize(new_img, (100, 100))
        new_img = np.resize(new_img, (100, 100, 1))
        new_img = new_img[np.newaxis, :, :, :]
        img_list = np.concatenate((img_list, new_img))

    for k in range(batch_size) :
        list_tmp = (list[k][1]/640.0, list[k][2]/480.0, list[k][3]/640.0, list[k][4]/480.0)
        label = np.vstack((label, list_tmp))

    label = label[1:][:]
    return (img_list, label)




img = cv2.imread('dataset/0.jpg')

X = tf.placeholder(tf.float32, [None, img_height, img_width, 1])
Y = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable('W1',[3, 3, 1, 32],dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
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
L8 = tf.nn.dropout(L8, keep_prob)

W9 = tf.get_variable('W9', [8, 8], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())
L9 = tf.matmul(L8, W9)
L9 = tf.nn.relu(L9)

W10 = tf.get_variable('W10', [8, 4], dtype=tf.float32, initializer=tf.keras.initializers.he_normal())

L10 = tf.matmul(L9, W10)




#cost = (tf.reduce_mean(tf.square(Y - L10))/2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L10, labels=Y))
#optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(cost)
optimizer = tf.train.AdadeltaOptimizer(0.0001).minimize(cost)

ckpt = tf.train.get_checkpoint_state('./model_save')

init = tf.global_variables_initializer()
sess = tf.Session()

saver = tf.train.Saver(tf.global_variables())

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) :
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("reload")
else :
    sess.run(init)

batch_size = 21
total_batch = 12

for epoch in range(100000) :
    total_cost = 0

    for i in range(total_batch) :
        batch_xs, batch_ys = next_batch(i, batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y:batch_ys, keep_prob:0.5})
        #print(i)
        total_cost += cost_val

    print('Epoch : ', '%04d'%(epoch+1), 'avg. cost =','{:.4f}'.format(total_cost/total_batch))
    if epoch % 10 == 0 :
        save_path = saver.save(sess, "./model_save/eye_track_model.ckpt")
        print('saved')
    #if total_cost < 50.0 :
    #    save_path = saver.save(sess, "./model_save/eye_track_model.ckpt")
    #    print('cost_val is lower than 50')
    #    break

save_path = saver.save(sess, "./model_save/eye_track_model.ckpt")
print("Model saved in path: %s" % save_path)


