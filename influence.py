import tensorflow as tf
import numpy as np
from vectorify import Vectorify
from conjugate_gradient import conjugate_gradient
from utils import minibatch_run
from Hv_prod import Hv_prod


class Influence:

    def __init__(self, func, evalset, loss, trainset, input_ph, target_ph, scale, func_grads=None, loss_grads=None, initial_dampening=0.1e-3, cg_iters=10, normal_equation=False, vervose=False, minibatch_size=1000):
        # func            The function of which we want to compute the influecne
        # evalset         The dataset on which func should be evaluated
        # loss            The 
        # trainset        The 
        # input_ph        The 
        # target_ph       The 
        # scale           The 
        # func_grads      The 
        # loss_grads      The 
        # dampening       The 
        # cg_iters        The 
        # normal_equation The 
        # vervose         The 
        # minibatch_size  The 

        if func_grads == None:
            func_grads = tf.gradients(func, tf.trainable_variables(),
                             name="influence_func_grads")
        if loss_grads == None:
            loss_grads = tf.gradients(loss, tf.trainable_variables(),
                             name="influence_loss_grads")
        self.func_grads = [g for g in func_grads if g is not None]
        self.loss_grads = [g for g in loss_grads if g is not None]
        self.loss = loss
        self.func = func 
        self.input_ph = input_ph
        self.target_ph = target_ph
        self.evalset = evalset
        self.trainset = trainset
        self.sess = tf.get_default_session()
        self.initial_dampening = initial_dampening
        self.dampening = initial_dampening
        self.normal_equation = normal_equation
        self.minibatch_size = minibatch_size
        self.cg_iters = cg_iters 
        self.vervose = vervose
        self.scale = scale

        self.Hvp, self.vecs = Hv_prod(loss, loss_grads)

        self.s = None

    def Hv_f(self, v):
        def minibatch_feed_dict(a, b):
            hv_feed_dic = {self.input_ph: self.trainset.images[
                a:b], self.target_ph: self.trainset.labels[a:b]}
            for i in range(len(self.vecs)):
                v_i = v[i]
                hv_feed_dic[self.vecs[i]] = v_i
            return hv_feed_dic
        # compute the Hvp using the above feed_dict generator and add the dampening
        Hvp_np =  minibatch_run(self.Hvp, minibatch_feed_dict, len(self.trainset.labels))/self.scale
        return Hvp_np + Vectorify(v) * self.dampening

    def normal_Hv_f(self, v):
        return self.Hv_f(self.Hv_f(v))

    def of(self, z):
        z_influence, z_grads = self.of_and_g(z) 
        return z_influence

    def of_and_g(self, z):
        if self.s == None:
            raise Exception("Before computing the influence of z, s needs to be computed")
        feed_dict = {self.input_ph: z[0], self.target_ph: z[1]}
        grads_on = Vectorify(self.sess.run(self.loss_grads, feed_dict))
        return -grads_on.dot(self.s), grads_on.norm()

    def compute_s(self):
        self.dampening = self.initial_dampening
        self.evalset_func_grads = minibatch_run(self.func_grads, lambda a, b: {self.input_ph: self.evalset.images[
            a:b], self.target_ph: self.evalset.labels[a:b]}, end=len(self.evalset.labels))
        if not self.normal_equation:
            solution, self.cg_error = conjugate_gradient(
                self.Hv_f, self.evalset_func_grads, self.cg_iters, vervose=self.vervose)
        else:
            print("Warning: using the normal equations leads to numerical instability in CG")
            solution, self.cg_error = conjugate_gradient(self.normal_Hv_f, self.Hv_f(
                self.evalset_func_grads), self.cg_iters, vervose=self.vervose)
        self.s = solution* self.scale if solution else None

    def robust_compute_s(self):
        self.dampening = self.initial_dampening
        while True: 
            self.evalset_func_grads = minibatch_run(self.func_grads, lambda a, b: {self.input_ph: self.evalset.images[
                a:b], self.target_ph: self.evalset.labels[a:b]}, end=len(self.evalset.labels))
            if not self.normal_equation:
                solution, self.cg_error = conjugate_gradient(
                    self.Hv_f, self.evalset_func_grads, self.cg_iters, vervose=self.vervose)
            else:
                solution, self.cg_error = conjugate_gradient(self.normal_Hv_f, self.Hv_f(
                    self.evalset_func_grads), self.cg_iters, vervose=self.vervose)
            if solution == None:
                self.dampening *= 10 
                print("changing the dampening to:", self.dampening)
            else:
                break
        self.s =  solution* self.scale

    def save_s(self, filename):
        self.s.save(filename)

    def load_s(self, filename):
        self.s = Vectorify(filename)
        self.evalset_func_grads = minibatch_run(self.func_grads, lambda a, b: {self.input_ph: self.evalset.images[
            a:b], self.target_ph: self.evalset.labels[a:b]}, end=len(self.evalset.labels))


# ---------------------------------------------------------------------------------

    def rand_hvp(self, n=10):
        lst = []
        for i in range(n):
            rand_v = [np.random.normal(size=g.shape)
                      for g in self.grads_on_dataset]
            res = self.vhv(rand_v)
            lst.append(res)
        return lst

    def hvp(self, v):
        Hv = ladd(self.Hv_f(v), lprod(self.dampening, v))
        return Hv

    def r(self, v):
        return lsubtract(b, ladd(Ax_f(x), lprod(self.dampening, x)))

    def vhv(self, v):
        return ldot(v, self.hvp(v))

    def call_cg(self, ):
        self.H_inv_grad_on = conjugate_gradient(
            self.Hv_f, self.grads_on_dataset, 20, self.dampening)
