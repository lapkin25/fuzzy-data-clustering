# Множественная кусочно-постоянная регрессия с нечетким целевым функционалом

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from regression import sort_points, points_partition,\
    sum_of_squares_of_deviations_from_mean

# Принимает на вход границы диапазонов t и координаты точек x
#   t[j] - это граница между j-м и (j+1)-м диапазонами, j = 0..m-2
# Возвращает двумерный массив u[i][k] - мера принадлежности
#   i-й точки k-му диапазону
# u_{ik} = u_k(x_i) - линейная функция на каждом промежутке
#   между центрами диапазонов, u_k = 1 в центре диапазона и
#   u_k = 0 в центре соседнего диапазона
def compute_u(t, x):
    n = len(x)  # количество точек
    m = len(t) + 1  # количество диапазонов
    u = np.array([[0.0 for k in range(m)] for i in range(n)])
    x_min = min(x)
    x_max = max(x)
    a = [0 for k in range(m)]  # середины диапазонов
    a[0] = (x_min + t[0]) / 2
    a[m - 1] = (t[m - 2] + x_max) / 2
    for k in range(1, m - 1):
        a[k] = (t[k - 1] + t[k]) / 2
    for i in range(n):
        for k in range(m):
            if k == 0:
                if x[i] <= a[0]:
                    u_val = 1
                elif x[i] > a[0] and x[i] <= a[1]:
                    u_val = (a[1] - x[i]) / (a[1] - a[0])
                else:
                    if x_min <= t[0]:
                        u_val = 0
                    else:
                        u_val = 1  # штраф
            elif k == m - 1:
                if x[i] >= a[m - 1]:
                    u_val = 1
                elif x[i] >= a[m - 2] and x[i] < a[m - 1]:
                    u_val = (x[i] - a[m - 2]) / (a[m - 1] - a[m - 2])
                else:
                    if x_max >= t[m - 2]:
                        u_val = 0
                    else:
                        u_val = 1  # штраф
            else:  # 0 < k < m - 1
                if x[i] >= a[k - 1] and x[i] <= a[k]:
                    u_val = (x[i] - a[k - 1]) / (a[k] - a[k - 1])
                elif x[i] >= a[k] and x[i] <= a[k + 1]:
                    u_val = (a[k + 1] - x[i]) / (a[k + 1] - a[k])
                else:
                    if t[k - 1] <= t[k]:
                        u_val = 0
                    else:
                        u_val = 1  # штраф
            u[i, k] = u_val
    return u

#TODO потестировать функцию compute_u отдельно


# Возвращает массив c[k] - средние значения в диапазонах
def compute_c(u, y):
    n = len(y)  # количество точек
    m = u.shape[1]  # количество диапазонов

    c = [0.0 for k in range(m)]
    for k in range(m):
        s = np.sum(u[:, k])
        if s == 0:
            c[k] = 0.0
        else:
            c[k] = np.dot(u[:, k], y) / s

    return c


def sort_points_and_u(x_unsorted, y_unsorted, u_unsorted):
#    print("x0 =", x_unsorted)
#    print("y0 =", y_unsorted)
#    print("u0 =", u_unsorted)
    n = len(x_unsorted)
    assert(len(y_unsorted) == n)
    # сортируем точки по координате x
    ind_p = x_unsorted.argsort()  # находим перестановку индексов
    x = x_unsorted[ind_p]
    y = y_unsorted[ind_p]
    u = u_unsorted[ind_p, :]

#    print("x =", x)
#    print("y =", y)
#    print("u =", u)
    return x, y, u


# Находит оптимальное разбиение точек на диапазоны
#   при заданных мерах принадлежности u_{ik} i-й точки k-му диапазону
# Возвращает границы диапазонов t[0], t[1], ..., t[m-2]:
#   t[j] - это граница между j-м и (j+1)-м диапазонами
# Критерий оптимальности - минимум функционала
#   J = sum_k sum_i u_{ik} * (y_i - avg_k)^2
def weighted_points_partition_given_u_avg(x_unsorted, y_unsorted, m, u_unsorted, avg):
    x, y, u = sort_points_and_u(x_unsorted, y_unsorted, u_unsorted)
    n = len(x)

    # Подзадача с параметрами (i, k):
    #   разбить точки 0..i на k диапазонов так, чтобы
    #   минимизировать сумму квадратов отклонений y от среднего по диапазону
    # Решение подзадачи:
    #   f[i, k] - найденное значение минимума функционала
    #   c[i, k] - число точек в самом правом диапазоне
    # Здесь i = 0..n-1, k = 1..m
    f = np.array([[None for k in range(m + 1)] for i in range(n)])
    c = np.array([[None for k in range(m + 1)] for i in range(n)])

    # Инициализация: единственный диапазон (k = 1)
    # для i = 0..n-1:
    #   f[i, 1] = взвешенная сумма квадратов отклонений от среднего y с 0-й точки до i-й
    #   c[i, 1] = i + 1
    #   sum_u_y[i][k] = сумма u[i'][k] * y[i'] по i' от 0 до i
    #   sum_sq_y[i] = сумма y[0..i]^2
    #   sum_u[i][k] = сумма u[i'][k] по i' от 0 до i
    sum_u_y = [[0 for k in range(m)] for i in range(n)]
    sum_sq_y = [0 for i in range(n)]
    sum_u = [[0 for k in range(m)] for i in range(n)]
    for k in range(m):
        sum_u_y[0][k] = u[0][k] * y[0]
        sum_u[0][k] = u[0][k]
    sum_sq_y[0] = y[0] ** 2
    for i in range(1, n):
        sum_sq_y[i] = sum_sq_y[i - 1] + y[i] ** 2
        for k in range(m):
            sum_u_y[i][k] = sum_u_y[i - 1][k] + u[i][k] * y[i]
            sum_u[i][k] = sum_u[i - 1][k] + u[i][k]
    for i in range(n):
        f[i, 1] = sum_sq_y[i]
        for l in range(m):
            f[i, 1] -= 2 * sum_u_y[i][l] * avg[l]
            f[i, 1] += sum_u[i][l] * avg[l] ** 2
        #f[i, 1] = sum_of_squares_of_deviations_from_mean(y[:i+1])
        c[i, 1] = i + 1

    # для k = 1..m-1 для i = 0..n-1:
    #   имея решение подзадачи (i, k),
    #   пробуем улучшить решение подзадачи (i + j, k + 1),
    #   добавив (k + 1)-й диапазон из j точек с (i+1)-й до (i+j)-й (j = 0..n-i-1)
    for k in range(1, m):
        for i in range(n):
            # решение подзадачи (i, k) найдено, результаты хранятся в f[i, k] и c[i, k]
            for j in range(n - i):
                if j == 0:
                    weighted_sum_sq_mean_dev = 0
                else:
                    weighted_sum_sq_mean_dev = sum_sq_y[i + j] - sum_sq_y[i]
                    for l in range(m):
                        weighted_sum_sq_mean_dev -=\
                            2 * (sum_u_y[i + j][l] - sum_u_y[i][l]) * avg[l]
                        weighted_sum_sq_mean_dev +=\
                            (sum_u[i + j][l] - sum_u[i][l]) * avg[l] ** 2
                f_new = f[i, k] + weighted_sum_sq_mean_dev
                #f_new = f[i, k] + sum_of_squares_of_deviations_from_mean(y[i+1:i+j+1])
                if f[i + j, k + 1] is None or f_new < f[i + j, k + 1]:
                    f[i + j, k + 1] = f_new
                    c[i + j, k + 1] = j

    t = [None for k in range(m - 1)]
    # для k от m-1 до 1 (с шагом -1):
    #   находим левую границу (x-координату) k-го диапазона - t[k-1]
    #   (индексация диапазонов с нуля)
    j = n - 1  # индекс последней точки k-го диапазона
    for k in range(m - 1, 0, -1):
        # c[j, k + 1] - число точек в k-м диапазоне
        # j - c[j, k + 1] + 1 - индекс первой точки k-го диапазона
        if j == -1 or c[j, k + 1] == 0:
            # редкий случай, когда k-й диапазон не содержит точек
            if k == m - 1:
                t[k - 1] = x[n - 1]  # ещё нужно прибавить положительное слагаемое
            else:
                t[k - 1] = t[k]
        else:
            ind_first = j - c[j, k + 1] + 1
            if ind_first == 0:
                t[k - 1] = x[ind_first]
            else:
                t[k - 1] = (x[ind_first] + x[ind_first - 1]) / 2
            j = ind_first - 1

    #print(t)
    #print("f = ", f[n - 1, m])
    return t


def points_partition_given_avg(x_unsorted, y_unsorted, m, avg):
    x, y = sort_points(x_unsorted, y_unsorted)
    n = len(x)

    # Подзадача с параметрами (i, k):
    #   разбить точки 0..i на k диапазонов так, чтобы
    #   минимизировать сумму квадратов отклонений y от среднего по диапазону
    # Решение подзадачи:
    #   f[i, k] - найденное значение минимума функционала
    #   c[i, k] - число точек в самом правом диапазоне
    # Здесь i = 0..n-1, k = 1..m
    f = np.array([[None for k in range(m + 1)] for i in range(n)])
    c = np.array([[None for k in range(m + 1)] for i in range(n)])

    # Инициализация: единственный диапазон (k = 1)
    # для i = 0..n-1:
    #   f[i, 1] = взвешенная сумма квадратов отклонений от среднего y с 0-й точки до i-й
    #   c[i, 1] = i + 1
    #   sum_u_y[i][k] = сумма u[i'][k] * y[i'] по i' от 0 до i
    #   sum_sq_y[i] = сумма y[0..i]^2
    #   sum_u[i][k] = сумма u[i'][k] по i' от 0 до i
    sum_y = [0 for i in range(n)]
    sum_sq_y = [0 for i in range(n)]
    sum_y[0] = y[0]
    sum_sq_y[0] = y[0] ** 2
    for i in range(1, n):
        sum_sq_y[i] = sum_sq_y[i - 1] + y[i] ** 2
        sum_y[i] = sum_y[i - 1] + y[i]
    for i in range(n):
        f[i, 1] = sum_sq_y[i] - 2 * avg[0] * sum_y[i] + (i + 1) * avg[0] ** 2
        #f[i, 1] = sum_of_squares_of_deviations_from_mean(y[:i+1])
        c[i, 1] = i + 1

    # для k = 1..m-1 для i = 0..n-1:
    #   имея решение подзадачи (i, k),
    #   пробуем улучшить решение подзадачи (i + j, k + 1),
    #   добавив (k + 1)-й диапазон из j точек с (i+1)-й до (i+j)-й (j = 0..n-i-1)
    for k in range(1, m):
        for i in range(n):
            # решение подзадачи (i, k) найдено, результаты хранятся в f[i, k] и c[i, k]
            for j in range(n - i):
                if j == 0:
                    sum_sq_mean_dev = 0
                else:
                    sum_sq_mean_dev = (sum_sq_y[i + j] - sum_sq_y[i])\
                        - 2 * avg[k] * (sum_y[i + j] - sum_y[i]) + j * avg[k] ** 2
                f_new = f[i, k] + sum_sq_mean_dev
                #f_new = f[i, k] + sum_of_squares_of_deviations_from_mean(y[i+1:i+j+1])
                if f[i + j, k + 1] is None or f_new < f[i + j, k + 1]:
                    f[i + j, k + 1] = f_new
                    c[i + j, k + 1] = j

    t = [None for k in range(m - 1)]
    # для k от m-1 до 1 (с шагом -1):
    #   находим левую границу (x-координату) k-го диапазона - t[k-1]
    #   (индексация диапазонов с нуля)
    j = n - 1  # индекс последней точки k-го диапазона
    for k in range(m - 1, 0, -1):
        # c[j, k + 1] - число точек в k-м диапазоне
        # j - c[j, k + 1] + 1 - индекс первой точки k-го диапазона
        if j == -1 or c[j, k + 1] == 0:
            # редкий случай, когда k-й диапазон не содержит точек
            if k == m - 1:
                t[k - 1] = x[n - 1]  # ещё нужно прибавить положительное слагаемое
            else:
                t[k - 1] = t[k]
        else:
            ind_first = j - c[j, k + 1] + 1
            if ind_first == 0:
                t[k - 1] = x[ind_first]
            else:
                t[k - 1] = (x[ind_first] + x[ind_first - 1]) / 2
            j = ind_first - 1

    #print(t)
    #print("f = ", f[n - 1, m])
    return t


# Возвращает средние значения для каждого диапазона,
#   взвешенную сумму квадратов отклонений и коэффициент детерминации
def fuzzy_partition_summary(x, y, u):
    #n = len(x)  # количество точек
    m = u.shape[1]  # количество диапазонов

    c = [0.0 for k in range(m)]
    for k in range(m):
        if np.sum(u[:, k]) == 0:
            c[k] = 0
        else:
            c[k] = np.dot(u[:, k], y) / np.sum(u[:, k])
    sum_squares = 0.0
    for k in range(m):
        sum_squares += np.dot(u[:, k], (y - c[k]) ** 2)
    R2 = 1 - sum_squares / sum_of_squares_of_deviations_from_mean(y)

    return c, sum_squares, R2


# Принимает на вход два массива координат точек
#   x_unsorted, y_unsorted и число диапазонов m
# Возвращает границы диапазонов t[0], t[1], ..., t[m-2]:
#   t[j] - это граница между j-м и (j+1)-м диапазонами
#   (диапазон включает левую границу и не включает правую)
# TODO добавить возвращаемый объект u
# Критерий оптимальности - минимум функционала
#   J = sum_k sum_i u_{ik} * (y_i - c_k)^2
#   где u_{ik} = u_k(x_i) - линейная функция на каждом промежутке
#   между центрами диапазонов, u_k = 1 в центре диапазона и
#   u_k = 0 в центре соседнего диапазона
# Оптимизируем J по t_j и c_k, причем считаем t_j равными середине
#   промежутка между соседними точками x_i
def fuzzy_points_partition(x_unsorted, y_unsorted, m):
    # вызываем функцию четкого разбиения, берем найденное разбиение
    #   в качестве начального приближения
    t0 = points_partition(x_unsorted, y_unsorted, m)
    def h(t):
        #t.sort() - вместо этого надо добавить штраф за нарушение порядка
        #print(t)
        _, J, _ = fuzzy_partition_summary(x_unsorted, y_unsorted, compute_u(t, x_unsorted))
       # print("  ", J)
        return J
        #return fuzzy_partition_summary(x_unsorted, y_unsorted, compute_u(t, x_unsorted))[1]
    res = minimize(h, t0)
    print("  t = ", res.x)
    return res.x

"""
    # возвращаем "четкое" разбиение в качестве оптимального,
    # а функционал от него будет вычисляться как нечеткий
    return t
"""
"""
    # пока разбиение не перестанет изменяться,
    #   находим меры принадлежности для текущего разбиения
    #   и вычисляем оптимальное нечеткое разбиение с новыми u_{ik}
    print(t)
    print("Оптимизируем нечеткое разбиение...")
    start_flag = True
    while True:
        u = compute_u(t, x_unsorted)
        c = compute_c(u, y_unsorted)
        #print(u)
        t = weighted_points_partition_given_u_avg(x_unsorted, y_unsorted, m, u, c)
        #print("t =", t)
        if not start_flag:
            J_old = J
        c, J, R2 = fuzzy_partition_summary(x_unsorted, y_unsorted, u)
        #print("c = ", c)
        print("  J =", J)
        if start_flag:
            start_flag = False
        else:
            if J_old == J:
                break
    print("Найденные границы:", t)
    return t
"""


def fuzzy_plot_points_partition(data_x, data_y, t, u):
    avg_y, _, _ = fuzzy_partition_summary(data_x, data_y, u)
    plt.plot(data_x, data_y, 'ro')
    min_x = np.min(data_x)
    max_x = np.max(data_x)
    min_y = np.min(data_y)
    max_y = np.max(data_y)
    for i in range(len(t)):
        plt.plot([t[i], t[i]], [min_y, max_y], 'g')
    for i in range(len(t) + 1):
        if i == 0:
            x1 = min_x
            x2 = t[i]
        elif i == len(t):
            x1 = t[i - 1]
            x2 = max_x
        else:
            x1 = t[i - 1]
            x2 = t[i]
        plt.plot([x1, x2], [avg_y[i], avg_y[i]], 'b', linestyle='dashed')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def fuzzy_plot_points_partition_coloured(data_x, data_y, t, u, cvalue):
    avg_y, _, _ = fuzzy_partition_summary(data_x, data_y, u)
    #plt.plot(data_x, data_y, 'ro', c=cvalue, cmap='gray')
    #plt.scatter(data_x, data_y, c=cvalue, cmap='Greys')
    plt.scatter(data_x, data_y, c=cvalue, cmap='Reds')
    min_x = np.min(data_x)
    max_x = np.max(data_x)
    min_y = np.min(data_y)
    max_y = np.max(data_y)
    for i in range(len(t)):
        plt.plot([t[i], t[i]], [min_y, max_y], 'g')
    for i in range(len(t) + 1):
        if i == 0:
            x1 = min_x
            x2 = t[i]
        elif i == len(t):
            x1 = t[i - 1]
            x2 = max_x
        else:
            x1 = t[i - 1]
            x2 = t[i]
        plt.plot([x1, x2], [avg_y[i], avg_y[i]], 'b', linestyle='dashed')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


"""
data_x = np.array([1, -1, 0, -2, 3, 5])
data_y = np.array([1, 3, 1, -1, 0, -2])
t = fuzzy_points_partition(data_x, data_y, 3)
u = compute_u(t, data_x)
fuzzy_plot_points_partition(data_x, data_y, t, u)
_, _, R2 = fuzzy_partition_summary(data_x, data_y, u)
print("R2 =", R2)
"""

#_, _, R2 = partition_summary(data_x, data_y, t)
#print("R2 =", R2)

