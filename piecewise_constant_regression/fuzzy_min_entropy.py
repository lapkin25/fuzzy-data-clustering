import numpy as np
from scipy.optimize import minimize
from correspondence_matrix import calc_reduced_correspondence_matrix_given_c, calc_reduced_correspondence_matrix
from optimal_partition import fuzzy_optimal_partition, calc_c_k, fuzzy_entropy_optimal_partition
from regression import points_partition


# попеременный алгоритм нахождения оптимальных весов и оптимального разбиения точек на диапазоны
def fuzzy_min_entropy(x, y, z, m, iter_num, w0, t0, simplified=False):
    n = x.shape[0]  # количество точек
    p = x.shape[1]  # количество признаков
    w = w0.copy()
    t = t0.copy()
    integral_x = np.zeros(n)
    for _ in range(iter_num):
        # находим наилучшее разбиение при заданных весах
        for i in range(n):
            integral_x[i] = np.dot(x[i, :], w)
        iter_num = 100
        lam = 0.01
        #t = fuzzy_optimal_partition(integral_x, y, m, t, iter_num, lam)
        t = fuzzy_entropy_optimal_partition(integral_x, y, z, m, t, iter_num, lam, simplified=simplified)
        c = [calc_c_k(t, k, integral_x, y) for k in range(m)]
        print("t =", t)

        # находим наилучшие веса при заданном разбиении
        def f(w):
            nonlocal integral_x
            #for i in range(n):
            #    integral_x[i] = np.dot(x[i, :], w)
            integral_x = np.dot(x, w)
            J, mat = calc_reduced_correspondence_matrix_given_c(integral_x, y, z, t, c, simplified=simplified)
            print(J)
            print(mat)
            return J

        res = minimize(f, w, method='Nelder-Mead', options={'maxiter': 1000})
        #res = minimize(f, w, tol=1e-3, options={'maxiter': 1000})
        #res = minimize(f, w, tol=1e-2, options={'maxiter': 30})
        w = res.x

    return w, t, c


# используется разбиение на диапазоны с помощью четкой кусочно-постоянной регрессии
def fuzzy_min_entropy_t_crisp(x, y, z, m, w0):
    n = x.shape[0]  # количество точек
    p = x.shape[1]  # количество признаков
    w = w0.copy()

    # находим наилучшие веса при найденном разбиении
    def f(w):
        integral_x = np.dot(x, w)
        t = points_partition(integral_x, y, m)

        # еще улучшаем разбиение с помощью нечеткого функционала
       # iter_num = 50
       # lam = 0.01
       # t = fuzzy_optimal_partition(integral_x, y, m, t, iter_num, lam)

        # локально улучшаем разбиение с тем же функционалом
        iter_num = 250
        lam = 0.01
        t = fuzzy_entropy_optimal_partition(integral_x, y, z, m, t, iter_num, lam)

        print(t)
        J, mat = calc_reduced_correspondence_matrix(integral_x, y, z, t)
        print(J)
        print(mat)
        return J

    res = minimize(f, w, tol=1e-2, options={'maxiter': 300, 'disp': True}, method='Nelder-Mead')  # убрать method?
    w = res.x
    integral_x = np.dot(x, w)
    t = points_partition(integral_x, y, m)

    return w, t