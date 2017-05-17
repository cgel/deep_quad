import tensorflow as tf
from vectorify import Vectorify

class Model:
  def __init__(self, model_name, trainset, testset, sess):
    self.trainset = trainset
    self.testset = testset
    self.model_name = model_name
    self.sess = sess

    with tf.name_scope("net"):
      self.input_ph = tf.placeholder(tf.float32, [None, 784])
      if model_name == "convnet":
        self.y, weights = self.convnet(self.input_ph)
      elif model_name == "linear":
        self.y, weights = self.linear_model(self.input_ph)
      elif model_name == "two_layer":
        self.y, weights = self.two_layer_model(self.input_ph)
      else:
        Exception("This model does not exist")

    # Define loss and optimizer
    with tf.name_scope("loss"):
      self.y_ = tf.placeholder(tf.float32, [None, 10], name="y_target")
      batch_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y)
      regularization = tf.reduce_sum( [tf.nn.l2_loss(w) for w in weights])
      self.cross_entropy = tf.reduce_sum(batch_loss) + regularization * 0.001

    lr = tf.Variable(0.1)
    self.learning_rate_ph = tf.placeholder(tf.float32, ())
    self.change_lr = tf.assign(lr, self.learning_rate_ph)
    opt = tf.train.AdamOptimizer(lr)
    self.grads_and_vars = opt.compute_gradients(self.cross_entropy)
    self.grads = [g for g,v in self.grads_and_vars if g is not None] 
    self.train_step = opt.apply_gradients(self.grads_and_vars)

    # Test trained model
    with tf.name_scope("accuracy"):
      correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

 
  def update(self, n, learning_rate=0.1e-4):
      self.sess.run(self.change_lr, {self.learning_rate_ph:learning_rate})
      for _ in range(n):
          batch_xs, batch_ys = self.trainset.next_batch(500)
          self.sess.run(self.train_step, feed_dict={self.input_ph: batch_xs, self.y_: batch_ys})

  def train(self, vervose = 0):
    if self.model_name == "convnet":
      if vervose > 0:
        self.report()
      learning_rate = 1e-4
      for i in range(1, 21):
          if i%5 == 0:
            learning_rate = learning_rate/10.
            if vervose > 0:
              print("New learning rate of:", learning_rate)
          if vervose > 0:
            print(" --- ", i, " --- ")    
          self.update(2000, learning_rate)
          if vervose > 0:
            self.report()
      print("Done training")
    else:
      Exception("A training remige for this model does not exist")

  def test(self, dataset):
    test_acc, test_loss = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={input_ph: dataset.test.images,
                                    y_: dataset.test.labels})
    print("test acc", test_acc)
    print("test loss", test_loss)
    
  def testset_loss(self, ):
    return self.evaluate_on(self.testset)

  def evaluate_on(self, dataset):
    return self.sess.run(self.cross_entropy, feed_dict={self.input_ph: dataset.images, self.y_: dataset.labels})
 
  def report(self):
    test_feed_dic = {self.input_ph: self.testset.images, self.y_: self.testset.labels}
    
    batch_xs, batch_ys = self.trainset.next_batch(self.testset.images.shape[0])
    train_feed_dic = {self.input_ph:batch_xs, self.y_:batch_ys}

    acc, gs_np, loss = self.sess.run([self.accuracy, self.grads, self.cross_entropy], feed_dict=train_feed_dic)
    print("train:")
    print("  grad norm", Vectorify(gs_np).norm()/len(train_feed_dic[self.y_] ) )
    print("  loss", loss/len(train_feed_dic[self.y_]))
    print("  accuracy", acc)
    acc, gs_np, loss = self.sess.run([self.accuracy, self.grads, self.cross_entropy], feed_dict=test_feed_dic)
    print("test:")
    print("  grad norm", Vectorify(gs_np).norm()/len(test_feed_dic[self.y_] ) )
    print("  loss", loss/len(test_feed_dic[self.y_]))
    print("  accuracy", acc)

  # The models:

  def convnet(self, x):
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
  
  def linear_model(self, x):
    W = weight_variable([784, 10])
    b = bias_variable([10])
  
    y = tf.matmul(x, W) + b
  
    weights = [W, b]
    return y, weights
  
  def two_layer_model(self, x):
    W1 = weight_variable([784, 500])
    b1 = bias_variable([500])
    h = tf.tanh(tf.matmul(x, W1) + b1)
  
    W2 = weight_variable([500, 10])
    b2 = bias_variable([10])
    y = tf.matmul(h, W2) + b2
  
    weights = [W1, b1, W2, b2]
    return y, weights
  
  
# General tf helper funcs
  
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
  
  
