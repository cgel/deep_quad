import tensorflow as tf
import numpy as np
from vectorify import Vectorify


def Hv_prod(x, grads=None, name="Hessian_vector_product"):
    # extend the graph to compute Hv products
    with tf.name_scope(name):
        if grads == None:
            grads = tf.gradients(x, tf.trainable_variables(),
                             name="grads")
        grads = [g for g in grads if g is not None]
        # vector placeholder (a vector)
        with tf.name_scope("v"):
            vecs = [tf.placeholder(tf.float32, grads[i].get_shape(), name="v" + str(i)+"_ph") for i in range(len(grads))]
        # gradient vector product (a scalar)
        with tf.name_scope("grads_v_product"):
            tt = 1
            gvp = [tf.reduce_sum(grads[i] * vecs[i], name="gvp"+str(i) ) for i in range(len(grads))]
        # Hessian vector product (a vector)
        Hvp = [hvp for hvp in tf.gradients(
            gvp, tf.global_variables(), name="second_order_grads") if hvp is not None]
        return Hvp, vecs
