import numpy as np
from vectorify import Vectorify
import tensorflow as tf


def corrupt_mnist(mnist, corrupt_prob):
    assert corrupt_prob >= 0 and corrupt_prob <= 1, "this is not a valid corruption probability"
    size = len(mnist.train.labels)
    float_mask = np.random.random([size])
    mnist.train.corrupted_mask = np.less(float_mask, corrupt_prob)
    mnist.train.original_labels = mnist.train.labels.copy()
    for i in range(size):
        if mnist.train.corrupted_mask[i] == True:
            corrupted_label = np.random.randint(9)
            if np.argmax(mnist.train.labels[i]) == corrupted_label:
                corrupted_label = 9
            #corrupted_label = 0

            mnist.train.labels[i] = np.zeros([10])
            mnist.train.labels[i][corrupted_label] = 1
    return mnist


def minibatch_run(ops, feed_dict_f, end, minibatch_size=256, sess=None):
    if sess == None:
        sess=tf.get_default_session()
    a = 0
    b = minibatch_size
    feed_dict = feed_dict_f(a, b)
    res = Vectorify(sess.run(ops, feed_dict))
    while b < end:
        a = b
        b = min(a + minibatch_size, end)
        feed_dict = feed_dict_f(a, b)
        res += Vectorify(sess.run(ops, feed_dict))
    if res.size == 1:
        res = res.data[0]  
    return res


class Dataset:
    def __init__(self, data):
        self.images = data[0]
        self.labels = data[1]
        self.a = 0 
        self.b = None 
        
    def next_batch(self, batch_size):
        if self.b == None or self.b == len(self.labels):
            self.a = 0
            self.b = batch_size
        else:
            self.a = self.b
            self.b = min(self.a + batch_size, len(self.labels) )
            
        return (self.images[self.a:self.b], self.labels[self.a:self.b])



def leave_one_out_dataset(dataset, i):
    return Dataset( (np.delete(dataset.images, i, axis=0), np.delete(dataset.labels, i, axis=0)) )
