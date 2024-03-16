import math
import numpy as np
from optimal_partition import calc_c_k, calc_u_k_given_a, calc_u_k
import statsmodels.api as sm


def calc_correspondence_matrix(x, y, t):
    data_size = len(y)  # число точек
    m = len(t) + 1  # число интервалов
    c = [calc_c_k(t, k, x, y) for k in range(m)]
    #print(c)
    x_min = min(x)
    x_max = max(x)
    u = np.zeros((data_size, m))
    v = np.zeros((data_size, m))
    for i in range(data_size):
        for k in range(m):
            u[i, k] = calc_u_k(t, k, x[i], x_min, x_max)
        for k in range(m):
            v[i, k] = calc_u_k_given_a(k, y[i], c)
    #import sys
    #np.set_printoptions(threshold=sys.maxsize)
    #print(u)
    #print(v)
    mat = np.zeros((m, m))
    for k in range(m):
        for j in range(m):
            mat[k, j] = np.dot(u[:, k], v[:, j]) / np.sum(u[:, k])
    J = 0  # заглушка
    return J, mat


# Принимает на вход:
#   двумерный массив x (x[i, k] - k-я компонента i-й точки)
#   одномерный массив y (y[i] - координата i-й точки)
#   веса (u[i] - вес i-й точки)
# Возвращает два объекта:
#   массив коэффициентов при признаках
#   и число - свободный член
def calc_weighted_regression(x, y, u):
    if np.sum(u != 0) < 5:
        return np.zeros(x.shape[1]), np.mean(y[u != 0])

    x1 = x[u != 0]
    y1 = y[u != 0]
    u1 = u[u != 0]
    x1 = sm.add_constant(x1)

    reg_weighted = sm.WLS(y1, x1, weights=u1)
    reg_weighted_fitted = reg_weighted.fit()
    #print(reg_weighted_fitted.summary())

    params = reg_weighted_fitted.params

    return params[1:], params[0]


def calc_reduced_correspondence_matrix_given_c(x, y, z, t, c, verbose=False, return_s = False):
    data_size = len(y)  # число точек
    m = len(t) + 1  # число интервалов
    # c = [calc_c_k(t, k, x, y) for k in range(m)]

    x_min = min(x)
    x_max = max(x)
    u = np.zeros((data_size, m))
    for i in range(data_size):
        for k in range(m):
            u[i, k] = calc_u_k(t, k, x[i], x_min, x_max)

    mat = np.zeros((m, m))
    v = np.zeros((data_size, m))
    min_yr = None
    max_yr = None
    for k in range(m):
        w, w0 = calc_weighted_regression(z, y, u[:, k])
        if verbose:
            print("k =", k)
            print("w =", w)
            print("w0 =", w0)
        yr = y - (np.dot(z, w) + w0) + c[k]  # приведенный KPI
        if min_yr is None or np.min(yr) < min_yr:
            min_yr = np.min(yr)
        if max_yr is None or np.max(yr) > max_yr:
            max_yr = np.max(yr)
        for i in range(data_size):
            for j in range(m):
                v[i, j] = calc_u_k_given_a(j, yr[i], c)
        for j in range(m):
            if np.sum(u[:, k]) != 0:
                mat[k, j] = np.dot(u[:, k], v[:, j]) / np.sum(u[:, k])
    #print("min_yr =", min_yr, " max_yr =", max_yr)

    # нечеткий аналог расстояния Кульбака-Лейблера
    J = 0
    s = np.zeros(m)
    for k in range(m):
        s[k] = np.sum(u[:, k])
    # J = - sum_k ( sum_i u[i, k] * log( sum_i (u[i, k] * v[i, k]) / sum_i u[i, k]  ) )
    for k in range(m):
        if s[k] != 0:  # and mat[k, k] != 0:
            J -= s[k] * math.log(mat[k, k])

    if return_s:
        return J, mat, s
    else:
        return J, mat


def calc_reduced_correspondence_matrix(x, y, z, t, verbose=False, return_s = False):
    m = len(t) + 1  # число интервалов
    c = [calc_c_k(t, k, x, y) for k in range(m)]
    return calc_reduced_correspondence_matrix_given_c(x, y, z, t, c, verbose, return_s)
