import tensorflow as tf
import numpy as np
from vectorify import Vectorify
from conjugate_gradient import conjugate_gradient


class Influence:

    def __init__(self, loss, input_ph, target_ph, testset, trainset, grads=None, dampening=0.1e-3, cg_iters=10, normal_equation=False, vervose=False, minibatch_size=1000):
        # influence on  computed from dataset
        # grads:     gradients of "on" if they have already been computed. If
        # grads=None they will be recomputed.

        self.grads = grads
        self.loss = loss
        self.dampening = dampening
        self.input_ph = input_ph
        self.target_ph = target_ph
        self.testset = testset
        self.trainset = trainset
        self.minibatch_size = minibatch_size
        self.sess = tf.get_default_session()

        self.grad_loss_testset = minibatch_run(grads, lambda a, b: {input_ph: testset[0][
            a:b], target_ph: testset[1][a:b]}, end=len(testset[1]))

        self.Hvp = Hv_prod(loss, grads)

        def Hv_f(v):
            def minibatch_feed_dict(a, b):
                hv_feed_dic = {input_ph: trainset[0][
                    a:b], target_ph: trainset[1][a:b]}
                for i in range(len(self.vecs)):
                    hv_feed_dic[self.vecs[i]] = v[i]
                return hv_feed_dic
            return minibatch_run(Hvp, hv_feed_dic, len(trainset[1])) + v * self.dampening

        self.Hv_f = Hv_f

        if normal_equation == False:
            self.s = conjugate_gradient(
                Hv_f, self.grad_loss_testset, cg_iters, vervose=vervose)
        else:
            print(
                "Warning: using the normal equations leads to numerical instability in CG")

            def normal_Hv_f(v):
                return Hv_f(Hv_f(v))
            self.s = conjugate_gradient(normal_Hv_f, Hv_f(
                self.grad_loss_testset), cg_iters, vervose=vervose)

    def of(self, z):
        feed_dict = {self.input_ph: z[0], self.target_ph: z[1]}
        grads_on = self.sess.run(self.grads, feed_dict)
        return -ldot(grads_on, self.s)

    def of_and_g(self, z):
        feed_dict = {self.input_ph: z[0], self.target_ph: z[1]}
        grads_on = self.sess.run(self.grads, feed_dict)
        return -ldot(grads_on, self.s), lnorm(grads_on)

    def recompute_s(self, cg_iters=None, dampening=None, vervose=1):
        if cg_iters == None:
            cg_iters = self.cg_iters
        if dampening == None:
            dampening = self.dampening

        hv_feed_dic = {self.input_ph: self.trainset[
            0], self.target_ph: self.trainset[1]}

        def Hv_f(v):
            for i in range(len(self.vecs)):
                hv_feed_dic[self.vecs[i]] = v[i]
            return ladd(self.sess.run(self.Hvp, hv_feed_dic), lprod(dampening, v))

        self.s = conjugate_gradient(
            Hv_f, self.grad_loss_testset, cg_iters, vervose=vervose)

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
