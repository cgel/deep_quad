import tensorflow as tf
import numpy as np

def ladd(a,b):
    return list(map(np.add, a,b))

def lsubtract(a,b):
    return list(map(np.subtract, a,b))

def ldot(a,b):
    prods = lprod(a,b)
    return sum(list(map(np.sum, prods)))

def lnorm(a):
    aa = lprod(a,a)
    return sum(list(map(np.sum, aa)))**0.5
    
def lprod(k, a):
    if type(k) == type(0.) or type(k) == np.float_:
        return list(map(lambda x: np.multiply(x,k), a))
    else:
        return list(map(np.multiply, a, k))

def ldiv(a, k):
    if type(k) == type(0.) or type(k) == np.float_:
        return list(map(lambda x: np.divide(x,k), a))
    else:
        return list(map(np.divide, a, k))

def lcopy(a):
    return [x.copy() for x in a]

def lset(a, c):
    assert type(c) == type(0.)
    for i in range(len(a)):
        a[i].fill(c)
    return a

def lsize(a):
    size = 0
    for v in a:
        size += v.size
    return size

def conjugate_gradient(Ax_f, b, iters, vervose=False):
    global d, x, r, alpha, beta, r_new, Ad, dad
    #x = lset( lcopy(b), 0.)
    x = [np.random.normal(size=v.shape) for v in b]
    xx = Ax_f(x)
    r = lsubtract(b, xx)
    d = r
    if vervose:
        print("Error:", lnorm(lsubtract(Ax_f(x), b)))
    for i in range(iters):    
        Ad = Ax_f(d)
        Ad_scaling = lnorm(Ad)/lnorm(d)
        dad = ldot(d,Ad)
        assert dad >= 0., "Upsi dupsi! d.TAd=%.5f so Ad is not possitive definite but CG needs it to be :("%dad
        alpha = ldot(r,r)/dad
        x = ladd(x, lprod(alpha, d))
        r_new = lsubtract(r, lprod(alpha, Ad))
        beta = ldot(r_new, r_new)/ldot(r,r)
        d = ladd(r_new, lprod(beta, d))
        r = r_new
        if vervose:
            print("- iter:", i)
            print("Error:", lnorm(lsubtract(Ax_f(x), b)))
        if vervose > 1:
            print("dad:", dad)
            print("alpha:", alpha)
            print("beta:", beta)
            print("Ad scaling:", Ad_scaling)
            print("d norm:", lnorm(d))
    return x


class Influence:
    def __init__(self, loss, input_ph, target_ph, testset, trainset, grads=None, dampening=0.1e-7, cg_iters=10, normal_equation=False, vervose=False, minibatch_size=1000):
        # influence on  computed from dataset
        #   grads:     gradients of "on" if they have already been computed. If grads=None they will be recomputed.
        
        if grads==None:
            grads = [g for g in tf.gradients(loss, tf.global_variables(), name="grads_for_influence") if g is not None]
        self.grads = grads
        self.loss = loss
        self.dampening = dampening 
        self.input_ph = input_ph
        self.target_ph = target_ph
        self.testset = testset
        self.vervose = vervose
        self.trainset = trainset
        self.minibatch_size = minibatch_size
        self.sess = tf.get_default_session()
        
        # extend the graph to compute Hv products
        with tf.name_scope("Hvp"):
            with tf.name_scope("v"):
                self.vecs = []
                for i in range(len(grads)):
                    self.vecs.append(tf.placeholder(tf.float32, grads[i].get_shape(), name="v"+str(i)+"_ph"))
            # gradient vector product
            with tf.name_scope("gvp"):
                gvp = []
                for i in range(len(grads)):
                    gvp.append(tf.reduce_sum( grads[i] * self.vecs[i], name="gvp"+str(i)))
            Hvp = [hvp for hvp in tf.gradients(gvp, tf.global_variables(), name="second_order_grads") if hvp is not None]

        self.Hvp = Hvp
        # precompute H_inv_grad_on        
        # using a single batch (should break down into minibatches the task for big datasets)  
        self.grad_loss_testset = self.sess.run(grads, feed_dict={input_ph:testset[0], target_ph:testset[1]})
        #hv_feed_dic = {input_ph:trainset[0], target_ph:trainset[1]}

        def Hv_f(v):
            hv_feed_dic = {}
            for i in range(len(self.vecs)):
                hv_feed_dic[self.vecs[i]] = v[i]

            a = 0 
            b = self.minibatch_size
            hv_feed_dic[input_ph] = trainset[0][a:b]
            hv_feed_dic[target_ph] = trainset[1][a:b]
            res = self.sess.run(Hvp, hv_feed_dic)
            while b < len(self.trainset[0]):
                a = b 
                b = min(a + self.minibatch_size, len(self.trainset[0]))
                hv_feed_dic[input_ph] = trainset[0][a:b]
                hv_feed_dic[target_ph] = trainset[1][a:b]
                minibatch_res = self.sess.run(Hvp, hv_feed_dic)
                #minibatch_res = ladd(self.sess.run(Hvp, hv_feed_dic), lprod(self.dampening, v))
                res = ladd(res, minibatch_res)
            #return res
            return ladd(res, lprod(self.dampening, v))
        self.Hv_f = Hv_f

        if normal_equation == False:
            self.s = conjugate_gradient(Hv_f, self.grad_loss_testset, cg_iters, vervose=vervose)
        else:
            print("Warning: using the normal equations leads to numerical instability in CG")
            def normal_Hv_f(v):
                return Hv_f(Hv_f(v))
            self.s = conjugate_gradient(normal_Hv_f, Hv_f(self.grad_loss_testset), cg_iters, vervose=vervose)
            

    def of(self, z):
        feed_dict = {self.input_ph: z[0], self.target_ph:z[1]}
        grads_on = self.sess.run(self.grads, feed_dict)
        return -ldot(grads_on, self.s)
        #return -ldot(grads_on, self.s) * (1/len(self.trainset[0]))

    def of_and_g(self, z):
        feed_dict = {self.input_ph: z[0], self.target_ph:z[1]}
        grads_on = self.sess.run(self.grads, feed_dict)
        return -ldot(grads_on, self.s), lnorm(grads_on)
        #return -ldot(grads_on, self.s) * (1/len(self.trainset[0])), lnorm(grads_on)


    def recompute_s(self, cg_iters=None, dampening=None, vervose=1):
        if cg_iters == None:
            cg_iters = self.cg_iters
        if dampening == None:
            dampening = self.dampening

        hv_feed_dic = {self.input_ph:self.trainset[0], self.target_ph:self.trainset[1]}
        def Hv_f(v):
            for i in range(len(self.vecs)):
                hv_feed_dic[self.vecs[i]] = v[i]
            return ladd(self.sess.run(self.Hvp, hv_feed_dic), lprod(dampening, v))
 
        self.s = conjugate_gradient(Hv_f, self.grad_loss_testset, cg_iters, vervose=vervose)
# ---------------------------------------------------------------------------------

    def power_iteration(self, iters=None, epsilon=None):
        assert iters!=None or epsilon!=None, "need at least one stopping criterion"
        u_x = [np.random.normal(size=g.shape) for g in self.grad_loss_testset]
        x = ldiv(lnorm(u_x), u_x)
        for i in range(iters):
            u_x = self.u_Hv_f(x)
            e_value = lnorm(u_x)
            new_x = ldiv(e_value, u_x)
            d = abs(ldot(new_x,x))
            print("i:", i)
            print("e_value:", e_value)
            print("d:", d)
            if epsilon and 1 -d < epsilon:
                break
            x = new_x
        return e_value

    def rand_hvp(self,n=10):
        lst = []
        for i in range(n):
            rand_v = [np.random.normal(size=g.shape) for g in self.grads_on_dataset]
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
        self.H_inv_grad_on = conjugate_gradient(self.Hv_f, self.grads_on_dataset, 20, self.dampening)


