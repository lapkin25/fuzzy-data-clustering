from optimal_partition import calc_c_k, calc_u_k_given_a
from correspondence_matrix import calc_reduced_correspondence_matrix


# по квадратной матрице условных вероятностей принадлежности
#   точек каждого из m классов к одному из m других классов
#   вычисляет такую же матрицу для обратного соответствия между классами
def invert_probabilities(mat):
    ...

    return inv_mat


# Вычисляет распределение значений (приведенного) KPI при условии
#   принадлежности точки каждому из интервалов компетентности
# Возвращает двумерный массив len(t) x num_intervals
#   условных вероятностей
def calc_distribution(x, y, z, t, min_y, max_y, num_intervals):
    m = len(t) + 1  # число интервалов
    c = [calc_c_k(t, k, x, y) for k in range(m)]
    mat = calc_reduced_correspondence_matrix(x, y, z, t)
    mat_inv = invert_probabilities(mat)
    delta = (max_y - min_y) / num_intervals
    for s in range(num_intervals):
        l = min_y + delta * s
        r = l + delta
        # нас интересует интервал (l, r) на оси y
        yc = (l + r) / 2
        # найдем меры принадлежности точки yc к интервалам на оси y
        u = [calc_u_k_given_a(k, yc, c) for k in range(m)]





