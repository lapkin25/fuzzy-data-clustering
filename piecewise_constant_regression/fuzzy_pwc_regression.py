# Модуль для нахождения оптимального нечеткого разбиения точек на диапазоны
#   вместе с оптимальными весовыми коэффициентами
# Целевой функционал: J = sum_k sum_i u_{ik} * (y_i - c_k)^2,
#   здесь u_{ik} - мера принадлежности i-й точки k-му диапазону,
#   определяемая как кусочно-линейная зависимость,
#   c_k - среднее значение величины y на k-м диапазоне


import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from fuzzy_multivariate_regression import fuzzy_partition_summary,\
    fuzzy_plot_points_partition
from optimal_partition import fuzzy_optimal_partition, calc_J
from regression import points_partition


# Принимает на вход данные x_{ik}, y_i и количество диапазонов m,
#   а также число итераций iter_num и начальное разбиение t0
# Возвращает веса w
def fuzzy_multivariate_pwc_regression(x_panel, y_unsorted, m, iter_num, w0, t0):
    n = x_panel.shape[0]  # количество точек
    p = x_panel.shape[1]  # количество признаков

    #reg = LinearRegression().fit(x_panel, y_unsorted)
    #w = reg.coef_  # начальное приближение
    w = w0.copy()
    t = t0.copy()
    x_unsorted = np.array([0.0 for i in range(n)])
    #start_flag = True
    for _ in range(iter_num):
        # находим наилучшее разбиение при заданных весах
        for i in range(n):
            x_unsorted[i] = np.dot(x_panel[i, :], w)
        #if start_flag:
            # берем четкое разбиение в качестве начального приближения
        #    t = points_partition(x_unsorted, y_unsorted, m)
        t = fuzzy_optimal_partition(x_unsorted, y_unsorted, m, t, 500, 0.02)

        print("t =", t)

        # находим наилучшие веса при заданном разбиении
        def f(w):
            nonlocal x_unsorted
            for i in range(n):
                x_unsorted[i] = np.dot(x_panel[i, :], w)
            J = calc_J(t, x_unsorted, y_unsorted)
            print(J)
            return J

        res = minimize(f, w)  #, method='Nelder-Mead')
        w = res.x
    return w, t
