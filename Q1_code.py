#################
# Imports, etc. #
#################

import pandas as pd
import tensorflow as tf
import numpy as np

#######################
# read & prepare data #
#######################

# read from file
df_train_orig = pd.read_csv('zip.train.gz', compression = 'gzip', sep=' ', header=None)
df_test_orig = pd.read_csv('zip.test.gz', compression = 'gzip', sep=' ', header=None)

# take only 2,3,5 digits data
df_train = df_train_orig[df_train_orig[0].isin([2,3,5])]
df_test = df_test_orig[df_test_orig[0].isin([2,3,5])]

# separate into y and x
df_train_y = df_train.iloc[:,:1]
df_train_x = df_train.iloc[:,1:257]

df_test_y = df_test.iloc[:,:1]
df_test_x = df_test.iloc[:,1:257]

# represent y as one-hot encoding vector and convert dataframes into numpy ndarrays

df_train_y['is_2'] = df_train_y.apply(lambda row: 1 if row[0] == 2 else 0, axis = 1)
df_train_y['is_3'] = df_train_y.apply(lambda row: 1 if row[0] == 3 else 0, axis = 1)
df_train_y['is_5'] = df_train_y.apply(lambda row: 1 if row[0] == 5 else 0, axis = 1)

df_test_y['is_2'] = df_test_y.apply(lambda row: 1 if row[0] == 2 else 0, axis = 1)
df_test_y['is_3'] = df_test_y.apply(lambda row: 1 if row[0] == 3 else 0, axis = 1)
df_test_y['is_5'] = df_test_y.apply(lambda row: 1 if row[0] == 5 else 0, axis = 1)

df_train_y.head(10)
df_test_y.head(10)

ndarray_train_y = df_train_y.iloc[:,1:4].as_matrix()
ndarray_train_x = df_train_x.as_matrix()

ndarray_test_y = df_test_y.iloc[:,1:4].as_matrix()
ndarray_test_x = df_test_x.as_matrix()

########################
########################
########################
#   BASIC NNET MODEL   #
########################
########################
########################

########################
# prepare models infra #
########################

x = tf.placeholder(tf.float32, [None, 256])
W = tf.Variable(tf.zeros([256, 3]))
b = tf.Variable(tf.zeros([3]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

###################
# train the model #
###################

basic_model_sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(20):

  index_from = 100*i
  index_until = 100*i+100
  batch_xs = ndarray_train_x[index_from: index_until, :]
  batch_ys = ndarray_train_y[index_from: index_until, :]
  basic_model_sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


######################
# evaluate the model #
######################

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print basic_model_sess.run([accuracy, confusion_matrix], feed_dict={x: ndarray_test_x, y_: ndarray_test_y})
print basic_model_sess.run([accuracy, confusion_matrix], feed_dict={x: ndarray_train_x, y_: ndarray_train_y})

basic_model_sess.close()


########################
########################
########################
#    DEEP NNET MODEL   #
########################
########################
########################

########################
# prepare models infra #
########################

x = tf.placeholder(tf.float32, [None, 256])

# 1st layer
W_layer1 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1))
b_layer1 = tf.Variable(tf.constant(0.1, shape = [128]))
h_layer1 = tf.nn.relu(tf.matmul(x,W_layer1) + b_layer1)

# 2nd layer
W_layer2 = tf.Variable(tf.truncated_normal([128, 16], stddev=0.1))
b_layer2 = tf.Variable(tf.constant(0.1, shape =[16]))
h_layer2 = tf.nn.relu(tf.matmul(h_layer1, W_layer2) + b_layer2)

# final layer
W_final_layer = tf.Variable(tf.truncated_normal([16,3], stddev=0.1))
b_final_layer = tf.Variable(tf.constant(0.1, shape = [3]))
y_deep = tf.nn.softmax(tf.matmul(h_layer2,W_final_layer) + b_final_layer)

y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_deep), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

###################
# train the model #
###################

Deep_model_sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y_deep,1), tf.argmax(y_,1))
confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y_deep,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for epoc in range (10):
    for i in range(20):

        index_from = 100*i
        index_until = 100*i+100
        batch_xs = ndarray_train_x[index_from: index_until, :]
        batch_ys = ndarray_train_y[index_from: index_until, :]

        if i % 4 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
          print('step %d, training accuracy %g' % (i, train_accuracy))

        Deep_model_sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: ndarray_test_x, y_: ndarray_test_y}))

print Deep_model_sess.run(confusion_matrix, feed_dict={x: ndarray_test_x, y_: ndarray_test_y})


######################
# evaluate the model #
######################


print Deep_model_sess.run([accuracy, confusion_matrix], feed_dict={x: ndarray_test_x, y_: ndarray_test_y})
print Deep_model_sess.run([accuracy, confusion_matrix], feed_dict={x: ndarray_train_x, y_: ndarray_train_y})

Deep_model_sess.close()


#################################
#################################
#################################
#    CONVOLUTIONAL NNET MODEL   #
#################################
#################################
#################################

########################
# prepare models infra #
########################

x = tf.placeholder(tf.float32, [None, 256])
y_ = tf.placeholder(tf.float32, [None, 3])

# weight initialization functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution and pooling functions
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 16, 16, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([4 * 4 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# readout layer
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


############################
# train and evaluate model #
############################

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y_,1), tf.argmax(y_conv,1))


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for epoc in range(15):
      for i in range(20):
        index_from = 100 * i
        index_until = 100 * i + 100
        batch_xs = ndarray_train_x[index_from: index_until, :]
        batch_ys = ndarray_train_y[index_from: index_until, :]


        if i % 4 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
          print('step %d, training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

      print('test accuracy %g' % accuracy.eval(feed_dict={x: ndarray_test_x, y_: ndarray_test_y}))

  print sess.run(confusion_matrix, feed_dict={x: ndarray_test_x, y_: ndarray_test_y})