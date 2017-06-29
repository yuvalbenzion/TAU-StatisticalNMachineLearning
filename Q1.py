#################
# Imports, etc. #
#################

import pandas as pd
import tensorflow as tf

#######################
# read & prepare data #
#######################

# read from file
#df_train_orig = pd.read_csv('zip.train.gz', compression = 'gzip', sep=' ', header=None)
#df_test_orig = pd.read_csv('zip.test.gz', compression = 'gzip', sep=' ', header=None)

url_train = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz'
url_test = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz'

df_train_orig = pd.read_csv(url_train, compression = 'gzip', sep=' ', header=None)
df_test_orig = pd.read_csv(url_test, compression = 'gzip', sep=' ', header=None)


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

#df_train_y.head(10)
#df_test_y.head(10)

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
y_ = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.zeros([256, 3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# define loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# define optimizer
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

###################
# train the model #
###################

basic_model_sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# optimize over batches of the training data
for i in range(20):

  index_from = 100*i
  index_until = 100*i+100
  batch_xs = ndarray_train_x[index_from: index_until, :]
  batch_ys = ndarray_train_y[index_from: index_until, :]
  basic_model_sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

######################
# evaluate the model #
######################

# define model estimators
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

# placeholders for inputs (Xs and ys)
x = tf.placeholder(tf.float32, [None, 256])
y_ = tf.placeholder(tf.float32, [None, 3])

# 1st layer definition
W_layer1 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1))
b_layer1 = tf.Variable(tf.constant(0.1, shape = [128]))
h_layer1 = tf.nn.relu(tf.matmul(x, W_layer1) + b_layer1)

# 2nd layer definition
W_layer2 = tf.Variable(tf.truncated_normal([128, 16], stddev=0.1))
b_layer2 = tf.Variable(tf.constant(0.1, shape =[16]))
h_layer2 = tf.nn.relu(tf.matmul(h_layer1, W_layer2) + b_layer2)

# final layer definition
W_final_layer = tf.Variable(tf.truncated_normal([16,3], stddev=0.1))
b_final_layer = tf.Variable(tf.constant(0.1, shape = [3]))
y_deep = tf.nn.softmax(tf.matmul(h_layer2,W_final_layer) + b_final_layer)

# define loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_deep), reduction_indices=[1]))
# define optimizer
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

###################
# train the model #
###################

Deep_model_sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# define model estimators
correct_prediction = tf.equal(tf.argmax(y_deep,1), tf.argmax(y_,1))
confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y_deep,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run training for several # of epocs, and in each epoc - train with batches of the train data
for epoc in range (10):
    for i in range(20):

        index_from = 100*i
        index_until = 100*i+100
        batch_xs = ndarray_train_x[index_from: index_until, :]
        batch_ys = ndarray_train_y[index_from: index_until, :]

        if i % 4 == 0: # print results
          train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
          print('step %d, training accuracy %g' % (i, train_accuracy))

        Deep_model_sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: ndarray_test_x, y_: ndarray_test_y}))

#print Deep_model_sess.run(confusion_matrix, feed_dict={x: ndarray_test_x, y_: ndarray_test_y})
#print Deep_model_sess.run(confusion_matrix, feed_dict={x: ndarray_train_x, y_: ndarray_train_y})

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

# placeholders for inputs (xs and ys)
x = tf.placeholder(tf.float32, [None, 256])
y_ = tf.placeholder(tf.float32, [None, 3])

# weight initialization functions (we'll use this more than once)
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) # a preferable way to initialize the W's - with truncated normal distribution
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution and pooling functions (we'll use this more than once)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# define first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 16, 16, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# define second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# apply activation function - RELU, and perform classic max pooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# define densely connected layer
W_fc1 = weight_variable([4 * 4 * 64, 1024])
b_fc1 = bias_variable([1024])

# flatten the shape we got from the max pool layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# define dropout layer
keep_prob = tf.placeholder(tf.float32) # define how much of the neurons we'd like to keep in each iteration in training
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


############################
# train and evaluate model #
############################

# define loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# define optinizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# define model estimators
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(y_conv,1), tf.argmax(y_,1))

# run the training
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

# run training for several # of epocs, and in each epoc - train with batches of the train data
  for epoc in range(15):
      for i in range(20):
        index_from = 100 * i
        index_until = 100 * i + 100
        batch_xs = ndarray_train_x[index_from: index_until, :]
        batch_ys = ndarray_train_y[index_from: index_until, :]


        if i % 4 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
          print('step %d, training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8})

      print('test accuracy %g' % accuracy.eval(feed_dict={x: ndarray_test_x, y_: ndarray_test_y, keep_prob: 1.0}))

    print sess.run([accuracy, confusion_matrix], feed_dict={x: ndarray_test_x, y_: ndarray_test_y, keep_prob: 1.0})
    print sess.run([accuracy, confusion_matrix], feed_dict={x: ndarray_train_x, y_: ndarray_train_y, keep_prob: 1.0})


