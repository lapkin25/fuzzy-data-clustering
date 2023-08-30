# Множественная кусочно-постоянная регрессия
# Для заданного множества многомерных точек x_{ij} и значений y_i
#   находим оптимальные весовые множители w_j такие, что
#   целевая функция J = sum_i (z_i - c_k)^2 минимальна
# Здесь z_i = sum_j w_j * x_{ij},
#   c_k - среднее значение y_i на k-м диапазоне изменения z
# (аргументами оптимизации являются веса w_j и границы диапазонов t_k)


import numpy as np
from sklearn.linear_model import LinearRegression
from regression import points_ordered_partition,\
    sum_of_squares_of_deviations_from_mean


def calc_J_given_partition(x_unsorted, y_unsorted, partition):
    m = len(partition)
    ind_p = x_unsorted.argsort()
    x = x_unsorted[ind_p]
    y = y_unsorted[ind_p]
    J = 0
    ind_start = 0  # индекс начала диапазона
    for k in range(m):
        ind_stop = ind_start + partition[k]  # индекс начала следующего диапазона
        J += sum_of_squares_of_deviations_from_mean(y[ind_start:ind_stop])
        ind_start = ind_stop
    return J


# Находит оптимальный весовой множитель
# i-я точка в момент времени t имеет координату x_i(t) = z_i + v_i * t
#   аргумент partition - это массив размеров диапазонов
# Функция возвращает t, при котором функционал кусочно-постоянной регрессии
#   для точек (x_i(t), y_i) при числе диапазонов m достигает минимума
def calc_optimal_weight_naive(v_unsorted, z_unsorted, y_unsorted, partition):
    n = len(y_unsorted)  # количество точек
    m = len(partition)  # количество диапазонов
    # сортируем точки по убыванию v[i]
    ind_p = (-v_unsorted).argsort()  # находим перестановку индексов
    v = v_unsorted[ind_p]
    z = z_unsorted[ind_p]
    y = y_unsorted[ind_p]
    x = np.array([0.0 for _ in range(n)])
    #partition = points_ordered_partition(y, m)  # стартовые размеры диапазонов
    J_opt = None
    t_opt = None
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            if v[i1] != v[i2]:
                t0 = (z[i1] - z[i2]) / (v[i2] - v[i1])
                for i in range(n):
                    x[i] = z[i] + v[i] * t0
                J = calc_J_given_partition(x, y, partition)
                if J_opt is None or J < J_opt or J == J_opt and t0 > t_opt:
                    J_opt = J
                    t_opt = t0
    print(J_opt)
    return t_opt - 0.0001  # чтобы не попасть в точку пересечения


# Находит оптимальный весовой множитель
# i-я точка в момент времени t имеет координату x_i(t) = z_i + v_i * t
#   аргумент partition - это массив размеров диапазонов
# Функция возвращает t, при котором функционал кусочно-постоянной регрессии
#   для точек (x_i(t), y_i) при числе диапазонов m достигает минимума
def calc_optimal_weight(v_unsorted, z_unsorted, y_unsorted, partition, guess):
    n = len(y_unsorted)  # количество точек
    m = len(partition)  # количество диапазонов
    # сортируем точки по убыванию v[i]
#    ind_p = (-v_unsorted).argsort()  # находим перестановку индексов
#    v = v_unsorted[ind_p]
#    z = z_unsorted[ind_p]
#    y = y_unsorted[ind_p]
    v = v_unsorted
    z = z_unsorted
    y = y_unsorted

    x = np.array([0.0 for _ in range(n)])
    #partition = points_ordered_partition(y, m)  # стартовые размеры диапазонов
    intersection_points = []
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            if v[i1] != v[i2]:
                t0 = (z[i1] - z[i2]) / (v[i2] - v[i1])
                intersection_points.append(t0)
    intersection_points.sort(reverse=True)
    t_opt = guess
    for i in range(n):
        x[i] = z[i] + v[i] * t_opt
  #  print("start x =", x)
    J_opt = calc_J_given_partition(x, y, partition)
  #  print("Start J = ", J_opt)
    for point_ind in range(len(intersection_points) - 1):
        t0 = (intersection_points[point_ind]
              + intersection_points[point_ind + 1]) / 2
        for i in range(n):
            x[i] = z[i] + v[i] * t0
        J = calc_J_given_partition(x, y, partition)
       # print("    t =", t0, "J =", J)
        if J < J_opt - 1e-6: #or J == J_opt and t0 > t_opt:
            J_opt = J
            t_opt = t0
    print("J = ", J_opt, "  at w =", t_opt)

  #  for i in range(n):
  #      x[i] = z[i] + v[i] * t_opt
  #  print("finish x =", x)

    return t_opt


# Принимает на вход данные x_{ik}, y_i и количество диапазонов m
# Возвращает веса w   ###, разбиение t и значение функционала J
def multivariate_pwc_regression(x_panel, y_unsorted, m):
    n = x_panel.shape[0]  # количество точек
    p = x_panel.shape[1]  # количество признаков

    reg = LinearRegression().fit(x_panel, y_unsorted)
    w0 = reg.coef_  # начальное приближение

    iter_num = 10  # количество итераций
    z = np.array([0.0 for i in range(n)])
    v = np.array([0.0 for i in range(n)])
    w = w0.copy()
    for i in range(n):
        z[i] = np.dot(x_panel[i, :], w)
    p0 = z.argsort()  # находим перестановку индексов
    # находим разбиение точек z_i на диапазоны (при текущих весах w_j)
    #   это будет начальное приближение для разбиения
    partition = points_ordered_partition(y_unsorted[p0], m)
    print(partition)
    for _ in range(iter_num):
        # оптимизируем функционал при заданных размерах диапазонов
        #   по очереди по каждой переменной w_j
        for j in range(p):
            print("j = ", j)
           # print("w =", w)
            for i in range(n):
                v[i] = x_panel[i, j]
                z[i] = np.dot(x_panel[i, :], w) - x_panel[i, j] * w[j]
            w[j] = calc_optimal_weight(v, z, y_unsorted, partition, w[j])
        # теперь обновляем разбиение
        for i in range(n):
            z[i] = np.dot(x_panel[i, :], w)
        p0 = z.argsort()  # находим перестановку индексов
        partition_new = points_ordered_partition(y_unsorted[p0], m)
        # TODO: если partition == partition_new, выйти из цикла
        partition = partition_new
        print(partition)

    return w
