import numpy as np
from optimal_partition import calc_c_k, calc_u_k_given_a, calc_u_k

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



# TODO: отдельная функция для приведенного KPI (reduced KPI)