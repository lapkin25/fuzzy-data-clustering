import numpy as np
from optimal_partition import calc_c_k, calc_u_k_given_a
from correspondence_matrix import calc_reduced_correspondence_matrix


# по квадратной матрице условных вероятностей принадлежности
#   точек каждого из m классов к одному из m других классов
#   вычисляет такую же матрицу для обратного соответствия между классами
# s - сумма мер принадлежности в каждом интервале по x
def invert_probabilities(mat, s):
    m = mat.shape[0]
    x_prob = s / np.sum(s)  # вероятности принадлежности каждому интервалу по x
    inv_mat = np.zeros((m, m))
    for i in range(m):  # i - диапазон KPI
        for j in range(m):  # j - диапазон компетенций
            # применяем формулу Байеса
            inv_mat[i, j] = mat[j, i] * x_prob[j]\
                            / sum([mat[k, i] * x_prob[k] for k in range(m)])
    return inv_mat


# Вычисляет распределение значений (приведенного) KPI при условии
#   принадлежности точки каждому из интервалов компетентности
# Возвращает двумерный массив m x num_intervals
#   условных вероятностей
def calc_distribution(x, y, z, t, min_y, max_y, num_intervals):
    m = len(t) + 1  # число интервалов
    c = [calc_c_k(t, k, x, y) for k in range(m)]
    _, mat, s = calc_reduced_correspondence_matrix(x, y, z, t, return_s=True)
    inv_mat = invert_probabilities(mat, s)
    #print(inv_mat)
    #print([np.sum(inv_mat[i, :]) for i in range(m)])
    delta = (max_y - min_y) / num_intervals
    # условные вероятности принадлежности значениям y при условии принадлежности интервалам x
    d = np.zeros((m, num_intervals))
    # условные вероятности принадлежности интервалам x при условии принадлежности значениям y
    p_x = np.zeros((m, num_intervals))
    for s in range(num_intervals):
        l = min_y + delta * s
        r = l + delta
        # нас интересует интервал (l, r) на оси y
        yc = (l + r) / 2
        # найдем меры принадлежности точки yc к интервалам на оси y
        u = [calc_u_k_given_a(k, yc, c) for k in range(m)]
        # находим вероятности отнесения данной точки к интервалам на оси x
        for i in range(m):
            p_x[i, s] = sum([inv_mat[k, i] * u[k] for k in range(m)])
    # находим условные вероятности отнесения точек к данному интервалу y
    #   при условии принадлежности определенным интервалам x
    # (считаем априорное распределение значений y равномерным)
    for i in range(m):
        print(np.sum(p_x[i, :]))
        for s in range(num_intervals):
            d[i, s] = p_x[i, s] / np.sum(p_x[i, :])
    return d
