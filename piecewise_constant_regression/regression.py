# Кусочно-постоянная регрессия

import numpy as np
import matplotlib.pyplot as plt

def sort_points(x_unsorted, y_unsorted):
    n = len(x_unsorted)
    assert(len(y_unsorted) == n)
    # сортируем точки по координате x
    ind_p = x_unsorted.argsort()  # находим перестановку индексов
    x = x_unsorted[ind_p]
    y = y_unsorted[ind_p]
    return x, y


def sum_of_squares_of_deviations_from_mean(a):
    if a.size == 0:
        return 0
    m = np.mean(a)
    return np.sum((a - m) ** 2)


# Принимает на вход y-координаты точек, x-координаты которых идут по
#   возрастанию, и число диапазонов m
# Возвращает массив количеств точек в каждом диапазоне,
#   минимизируя сумму квадратов отклонений y от среднего y в соответствующем
#   диапазоне
# Например, если m = 3, n = 10, а функция вернула [2, 3, 5], то
#   оптимальным будет разбиение 0..1, 2..4, 5..9
# Сложность алгоритма: O(n^2 * m)
def points_ordered_partition(y, m):
    n = len(y)

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
    #   f[i, 1] = сумма квадратов отклонений от среднего y с 0-й точки до i-й
    #   c[i, 1] = i + 1
    #   sum_y[i] = сумма y[0..i]
    #   sum_sq_y[i] = сумма y[0..i]^2
    sum_y = [0 for i in range(n)]
    sum_sq_y = [0 for i in range(n)]
    sum_y[0] = y[0]
    sum_sq_y[0] = y[0] ** 2
    for i in range(1, n):
        sum_y[i] = sum_y[i - 1] + y[i]
        sum_sq_y[i] = sum_sq_y[i - 1] + y[i] ** 2
    for i in range(n):
        f[i, 1] = sum_sq_y[i] - sum_y[i] ** 2 / (i + 1)
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
                    sum_sq_mean_dev = (sum_sq_y[i + j] - sum_sq_y[i]) - (sum_y[i + j] - sum_y[i]) ** 2 / j
                f_new = f[i, k] + sum_sq_mean_dev
                #f_new = f[i, k] + sum_of_squares_of_deviations_from_mean(y[i+1:i+j+1])
                if f[i + j, k + 1] is None or f_new < f[i + j, k + 1]:
                    f[i + j, k + 1] = f_new
                    c[i + j, k + 1] = j

    s = np.array([None for k in range(m)])
    # для k от m-1 до 0 (с шагом -1):
    #   находим число точек в k-м диапазоне - s[k-1]
    #   (индексация диапазонов с нуля)
    j = n - 1  # индекс последней точки k-го диапазона
    for k in range(m - 1, -1, -1):
        # c[j, k + 1] - число точек в k-м диапазоне
        # j - c[j, k + 1] + 1 - индекс первой точки k-го диапазона
        if j == -1:
            s[k] = 0
        else:
            s[k] = c[j, k + 1]
            j = j - c[j, k + 1]

    return s


# Принимает на вход два массива координат точек
#   x_unsorted, y_unsorted и число диапазонов m
# Возвращает границы диапазонов t[0], t[1], ..., t[m-2]:
#   t[j] - это граница между j-м и (j+1)-м диапазонами
#   (диапазон включает левую границу и не включает правую)
# Узкое место - когда несколько точек на границе диапазонов
#   имеют одинаковые координаты x - в этом случае они будут
#   отнесены к одному и тому же диапазону
def points_partition(x_unsorted, y_unsorted, m):
    x, y = sort_points(x_unsorted, y_unsorted)
    n = len(x)

    s = points_ordered_partition(y, m)

    t = [None for k in range(m - 1)]
    # для k от m-1 до 1 (с шагом -1):
    #   находим левую границу (x-координату) k-го диапазона - t[k-1]
    #   (индексация диапазонов с нуля)
    j = n - 1  # индекс последней точки k-го диапазона
    for k in range(m - 1, 0, -1):
        # s[k] - число точек в k-м диапазоне
        # j - s[k] + 1 - индекс первой точки k-го диапазона
        if j == -1 or s[k] == 0:
            # редкий случай, когда k-й диапазон не содержит точек
            if k == m - 1:
                t[k - 1] = x[n - 1]  # ещё нужно прибавить положительное слагаемое
            else:
                t[k - 1] = t[k]
        else:
            ind_first = j - s[k] + 1
            if ind_first == 0:
                t[k - 1] = x[ind_first]
            else:
                t[k - 1] = (x[ind_first] + x[ind_first - 1]) / 2
            j = ind_first - 1

    #print(t)
    return t


# Возвращает средние значения для каждого диапазона,
#   сумму квадратов отклонений и коэффициент детерминации
def partition_summary(x_unsorted, y_unsorted, t):
    x, y = sort_points(x_unsorted, y_unsorted)
    m = len(t) + 1  # количество диапазонов

    avg_y = []
    sum_squares = 0
    for k in range(m):
        # TODO: вынести повторяющийся код
        if k == 0:
            if y[x < t[0]].size == 0:
                sum_squares = 0
                avg = 0
            else:
                avg = np.mean(y[x < t[0]])
                sum_squares += sum_of_squares_of_deviations_from_mean(y[x < t[0]])
        elif k == m - 1:
            if y[x >= t[m - 2]].size == 0:
                sum_squares = 0
                avg = 0
            else:
                avg = np.mean(y[x >= t[m - 2]])
                sum_squares += sum_of_squares_of_deviations_from_mean(y[x >= t[m - 2]])
        else:
            if y[(x < t[k]) & (x >= t[k-1])].size == 0:
                sum_squares = 0
                avg = 0
            else:
                avg = np.mean(y[(x < t[k]) & (x >= t[k-1])])
                sum_squares += sum_of_squares_of_deviations_from_mean(y[(x < t[k]) & (x >= t[k-1])])
        avg_y.append(avg)

    R2 = 1 - sum_squares / sum_of_squares_of_deviations_from_mean(y)

    return avg_y, sum_squares, R2


def plot_points_partition(data_x, data_y, t):
    avg_y, _, _ = partition_summary(data_x, data_y, t)
    #print(avg_y)
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


#points_partition(np.array([1, 0, -2, 3, 5]), np.array([1, 3, 1, -1, -2]), 3)
"""
data_x = np.array([1, -1, 0, -2, 3, 5])
data_y = np.array([1, 3, 1, -1, 0, -2])
t = points_partition(data_x, data_y, 3)
plot_points_partition(data_x, data_y, t)
_, _, R2 = partition_summary(data_x, data_y, t)
print("R2 =", R2)
"""
