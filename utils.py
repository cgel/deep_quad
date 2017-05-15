import numpy as np
from vectorify import Vectorify


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


def minibatch_run(ops, feed_dict_f, end, minibatch_size=256, sess=tf.get_default_session()):
    a = 0
    b = minibatch_size
    feed_dict = feed_dict_f(a, b)
    res = Vectorify(sess.run(ops, feed_dict))
    while b < end:
        a = b
        b = min(a + minibatch_size, end)
        feed_dict = feed_dict_f(a, b)
        res += sess.run(ops, feed_dict)
    return res
