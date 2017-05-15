import tensorflow as tf
import numpy as np
from vectorify import Vectorify


def Hv_prod(x, grads=None):
    if grads == None:
        grads = tf.gradients(loss, tf.global_variables(),
                             name="grads_for_influence")
    grads = [g for g in grads if g is not None]

    # extend the graph to compute Hv products
    with tf.name_scope("Hv_prod"):
        # vector placeholder
        with tf.name_scope("v"):
            vecs = [tf.placeholder(tf.float32, grads[i].get_shape(
            ), name="v" + str(i) + "_ph") for i in range(len(grads))]
        # gradient vector product
        with tf.name_scope("gvp"):
            gvp = [tf.reduce_sum(
                grads[i] * self.vecs[i], name="gvp" + str(i)) for range(len(grads))]
        # Hessian vector product
        Hvp = [hvp for hvp in tf.gradients(
            gvp, tf.global_variables(), name="second_order_grads") if hvp is not None]
        return Hvp
