import numpy as np
from optimal_partition import fuzzy_optimal_partition
from scipy.optimize import minimize
from correspondence_matrix import calc_reduced_correspondence_matrix


def fuzzy_min_entropy(x, y, z, m, iter_num, w0, t0):
    n = x.shape[0]  # количество точек
    p = x.shape[1]  # количество признаков
    w = w0.copy()
    t = t0.copy()
    integral_x = np.zeros(n)
    for _ in range(iter_num):
        # находим наилучшее разбиение при заданных весах
        for i in range(n):
            integral_x[i] = np.dot(x[i, :], w)
        iter_num = 200
        lam = 0.01
        t = fuzzy_optimal_partition(integral_x, y, m, t, iter_num, lam)

        print("t =", t)

        # находим наилучшие веса при заданном разбиении
        def f(w):
            nonlocal integral_x
            for i in range(n):
                integral_x[i] = np.dot(x[i, :], w)
            J, _ = calc_reduced_correspondence_matrix(integral_x, y, z, t)
            print(J)
            return J

        res = minimize(f, w, method='Nelder-Mead', options={'maxiter': 1000})
        w = res.x

    return w, t