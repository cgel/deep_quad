from vectorify import Vectorify


def conjugate_gradient(Ax_f, b, iters, vervose=False):
    x = b.copy() * 0
    xx = Ax_f(x)
    r = b - xx
    d = r
    if vervose:
        print("Initial error:", (Ax_f(x) - b).norm())
    for i in range(iters):
        Ad = Ax_f(d)
        Ad_scaling = Ad.norm() / d.norm()
        dad = d.dot(Ad)
        if dad < 0.:
            return None, None
        alpha = r.dot(r) / dad
        x = x + d * alpha
        r_new = r - Ad * alpha
        beta = r_new.dot(r_new) / r.dot(r)
        d = r_new + d * beta
        r = r_new
        if vervose > 1:
            print("- iter:", i)
            print("Error:", (Ax_f(x) - b).norm())
        if vervose > 2:
            print("dad:", dad)
            print("alpha:", alpha)
            print("beta:", beta)
            print("Ad scaling:", Ad_scaling)
            print("d norm:", d.norm())
    final_err = (Ax_f(x) - b).norm()
    if vervose:
        print("Final error:", final_err)
    return x, final_err
