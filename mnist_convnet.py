import tensorflow as tf

def deepnn(x):
  nl = tf.nn.tanh
  #nl = tf.nn.elu
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  head = nl(conv2d(x_image, W_conv1, stride=2) + b_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  head = nl(conv2d(head, W_conv2, stride=1) + b_conv2)

  W_conv3 = weight_variable([5, 5, 64, 64])
  b_conv3 = bias_variable([64])
  head = nl(conv2d(head, W_conv3, stride=2) + b_conv3)

  W_conv4 = weight_variable([5, 5, 64, 64])
  b_conv4 = bias_variable([64])
  h_conv4 = nl(conv2d(head, W_conv4, stride=1) + b_conv4)

  W_conv5 = weight_variable([5, 5, 64, 64])
  b_conv5 = bias_variable([64])
  h_conv5 = nl(conv2d(head, W_conv5, stride=1) + b_conv5)

  W_conv6 = weight_variable([5, 5, 64, 64])
  b_conv6 = bias_variable([64])
  h_conv6 = nl(conv2d(head, W_conv6, stride=1) + b_conv6)

  head = tf.reshape(head, [-1, 7*7*64])

  W_fc1 = weight_variable([7 * 7 * 64, 512])
  b_fc1 = bias_variable([512])

  head = nl(tf.matmul(head, W_fc1) + b_fc1)

  W_fc2 = weight_variable([512, 512])
  b_fc2 = bias_variable([512])

  head = nl(tf.matmul(head, W_fc2) + b_fc2)

  W_fcf = weight_variable([512, 10])
  b_fcf = bias_variable([10])

  y_conv = tf.matmul(head, W_fcf) + b_fcf

  weights = []
  weights.append(W_conv1)
  weights.append(b_conv1)
  weights.append(W_conv2)
  weights.append(b_conv2)
  weights.append(W_conv3)
  weights.append(b_conv3)
  weights.append(W_conv4)
  weights.append(b_conv4)
  weights.append(W_conv5)
  weights.append(b_conv5)
  weights.append(W_conv6)
  weights.append(b_conv6)

  weights.append(W_fc1)
  weights.append(b_fc1)
  weights.append(W_fc2)
  weights.append(b_fc2)
  weights.append(W_fcf)
  weights.append(b_fcf)

  return y_conv, weights


def conv2d(x, W, stride=1):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


