# Модуль для нахождения оптимального нечеткого разбиения точек на диапазоны
# Целевой функционал: J = sum_k sum_i u_{ik} * (y_i - c_k)^2,
#   здесь u_{ik} - мера принадлежности i-й точки k-му диапазону,
#   определяемая как кусочно-линейная зависимость,
#   c_k - среднее значение величины y на k-м диапазоне

import random

# Принимает на вход границы диапазонов,
#   возвращает середины диапазонов
def calc_a(t, x_min, x_max):
    m = len(t) + 1  # количество диапазонов
    a = [0.0 for k in range(m)]  # середины диапазонов
    a[0] = (x_min + t[0]) / 2
    a[m - 1] = (t[m - 2] + x_max) / 2
    for k in range(1, m - 1):
        a[k] = (t[k - 1] + t[k]) / 2
    return a


def calc_u_k_given_a(k, x, a):
    m = len(a)  # количество диапазонов
    if k == 0:
        if x <= a[0]:
            u_val = 1
        elif x > a[0] and x <= a[1]:
            u_val = (a[1] - x) / (a[1] - a[0])
        else:
            u_val = 0
    elif k == m - 1:
        if x >= a[m - 1]:
            u_val = 1
        elif x >= a[m - 2] and x < a[m - 1]:
            u_val = (x - a[m - 2]) / (a[m - 1] - a[m - 2])
        else:
            u_val = 0
    else:  # 0 < k < m - 1
        if x >= a[k - 1] and x <= a[k]:
            u_val = (x - a[k - 1]) / (a[k] - a[k - 1])
        elif x >= a[k] and x <= a[k + 1]:
            u_val = (a[k + 1] - x) / (a[k + 1] - a[k])
        else:
            u_val = 0
    return u_val


# Принимает на вход (неубывающие) границы диапазонов t,
#   номер диапазона k (k in 0..m-1) и число x - координату точки
# Возвращает меру принадлежности точки x к k-му диапазону
def calc_u_k(t, k, x, x_min, x_max):
    a = calc_a(t, x_min, x_max)
    return calc_u_k_given_a(k, x, a)


# Принимает на вход (неубывающие) границы диапазонов t,
#   номер диапазона k (k in 0..m-1) и координаты точек (x_i, y_i)
# Возвращает среднее значение y на k-м диапазоне
def calc_c_k(t, k, x, y):
    n = len(y)  # количество точек
    s1 = 0
    s2 = 0
    x_min = min(x)
    x_max = max(x)
    for i in range(n):
        u_i_k = calc_u_k(t, k, x[i], x_min, x_max)
        s1 += u_i_k * y[i]
        s2 += u_i_k
    ans = s1 / s2 if s2 != 0 else 0
    return ans


# Принимает на вход t - границы разбиения на диапазоны и
#   x, y - массивы координат точек
# Возвращает значение целевой функции
def calc_J(t, x, y):
    n = len(x)  # количество точек
    m = len(t) + 1  # количество диапазонов
    x_min = min(x)
    x_max = max(x)
    s = 0.0
    for k in range(m):
        c_k = calc_c_k(t, k, x, y)
        for i in range(n):
            u_i_k = calc_u_k(t, k, x[i], x_min, x_max)
            s += u_i_k * (y[i] - c_k) ** 2
    return s

# Принимает на вход границы диапазонов t,
#   номер диапазона k, координату точки x и номер диапазона j
# Возвращает производную меры принадлежности u_{ik}
#   по a_j - середине j-го промежутка (t_j, t_{j+1})
def calc_duik_daj(t, k, x_i, j, x_min, x_max):
    m = len(t) + 1  # количество диапазонов
    a = calc_a(t, x_min, x_max)

    if k == 0:
        if j == 0:
            if x_i > a[0] and x_i < a[1]:
                d_val = (a[1] - x_i) / (a[1] - a[0]) ** 2
            else:
                d_val = 0
        elif j == 1:
            if x_i > a[0] and x_i < a[1]:
                d_val = (x_i - a[0]) / (a[1] - a[0]) ** 2
            else:
                d_val = 0
        else:
            d_val = 0
    elif k == m - 1:
        if j == m - 1:
            if x_i > a[m - 2] and x_i < a[m - 1]:
                d_val = (a[m - 2] - x_i) / (a[m - 1] - a[m - 2]) ** 2
            else:
                d_val = 0
        elif j == m - 2:
            if x_i > a[m - 2] and x_i < a[m - 1]:
                d_val = (x_i - a[m - 1]) / (a[m - 1] - a[m - 2]) ** 2
            else:
                d_val = 0
        else:
            d_val = 0
    else:  # 0 < k < m - 1
        if j == k:
            if x_i > a[k - 1] and x_i < a[k]:
                d_val = (a[k - 1] - x_i) / (a[k] - a[k - 1]) ** 2
            elif x_i > a[k] and x_i < a[k + 1]:
                d_val = (a[k + 1] - x_i) / (a[k + 1] - a[k]) ** 2
            else:
                d_val = 0
        elif j == k - 1:
            if x_i > a[k - 1] and x_i < a[k]:
                d_val = (x_i - a[k]) / (a[k] - a[k - 1]) ** 2
            else:
                d_val = 0
        elif j == k + 1:
            if x_i > a[k] and x_i < a[k + 1]:
                d_val = (x_i - a[k]) / (a[k + 1] - a[k]) ** 2
            else:
                d_val = 0
        else:
            d_val = 0

    return d_val


# Принимает на вход границы диапазонов t,
#   номер диапазона k, координату точки x и номер диапазона j
# Возвращает производную меры принадлежности u_{ik}
#   по t_j - границе между j-м и (j+1)-м диапазонами
def calc_duik_dtj(t, k, x_i, j, x_min, x_max):
    # d u_{ik} / d t_j = 1/2 * d u_{ik} / d a_j + 1/2 * d u_{ik} / d a_{j+1}
    duik_dtj = 1 / 2 * calc_duik_daj(t, k, x_i, j, x_min, x_max)\
               + 1 / 2 * calc_duik_daj(t, k, x_i, j + 1, x_min, x_max)
    return duik_dtj


# Принимает на вход t - границы разбиения на диапазоны и
#   координаты точек x, y
# Возвращает массив производных функционала по границам
def calc_derivatives_J_t(t, x, y):
    n = len(x)  # количество точек
    m = len(t) + 1  # количество диапазонов
    J_t = [0 for j in range(m - 1)]
    x_min = min(x)
    x_max = max(x)

    for k in range(m):
        c_k = calc_c_k(t, k, x, y)
        for i in range(n):
            for j in range(m - 1):
                duik_dtj = calc_duik_dtj(t, k, x[i], j, x_min, x_max)
                J_t[j] += c_k * (c_k - 2 * y[i]) * duik_dtj

    return J_t


def calc_derivatives_J_t_check(t, x, y):
    m = len(t) + 1  # количество диапазонов
    J_t = [0 for j in range(m - 1)]
    delta_t = 0.1

    J0 = calc_J(t, x, y)
    for j in range(m - 1):
        t[j] += delta_t
        J = calc_J(t, x, y)
        J_t[j] = (J - J0) / delta_t
        t[j] -= delta_t

    return J_t


# Принимает на вход координаты точек (x_i, y_i) и число диапазонов m,
#   а также начальное приближение t_0, число итераций iter_num
#   и параметр градиентного спуска lam
# Возвращает список границ диапазонов:
#   t[j] - граница между j-м и (j+1)-м диапазонами, j = 0..m-2
def fuzzy_optimal_partition(x, y, m, t0, iter_num, lam):
    t = t0.copy()
    J_pred = calc_J(t, x, y)
    t_pred = t.copy()
    cnt = 0
    for it in range(iter_num):
        # TODO: прокомментировать алгоритм
        J_t = calc_derivatives_J_t(t, x, y)

        #J_t_check = calc_derivatives_J_t_check(t, x, y)
        #print("J_t =", J_t)
        #print("J_t_check =", J_t_check)
        #for j in range(m - 1):

        j = random.choice(range(m - 1))
        t_j_new = t[j] - lam * J_t[j]
        if j > 0 and t_j_new < (t[j - 1] + t[j]) / 2:
            t_j_new = (t[j - 1] + t[j]) / 2
        elif j < m - 2 and t_j_new > (t[j] + t[j + 1]) / 2:
            t_j_new = (t[j] + t[j + 1]) / 2
        t[j] = t_j_new

        J = calc_J(t, x, y)
        if J < J_pred:
            cnt += 1
            if cnt == 5:
                lam *= 1.3
                cnt = 0
            t_pred = t.copy()
            J_pred = J
        else:
            lam /= 1.5
            cnt = 0
            t = t_pred.copy()
        print("   iteration", it)
        print("   t = ", t)
        print("   J =", J)
        #print("   lam =", lam)
    return t


from correspondence_matrix import calc_reduced_correspondence_matrix

def calc_derivatives_J_entropy_t(t, x, y, z, simplified=False):
    m = len(t) + 1  # количество диапазонов
    J_t = [0 for j in range(m - 1)]
    delta_t = 0.1

    J0, _ = calc_reduced_correspondence_matrix(x, y, z, t, simplified=simplified)
    for j in range(m - 1):
        t[j] += delta_t
        J, _ = calc_reduced_correspondence_matrix(x, y, z, t, simplified=simplified)
        J_t[j] = (J - J0) / delta_t
        t[j] -= delta_t

    return J_t

# Принимает на вход координаты точек (x_i, y_i) и число диапазонов m,
#   а также начальное приближение t_0, число итераций iter_num
#   и параметр градиентного спуска lam
# Возвращает список границ диапазонов:
#   t[j] - граница между j-м и (j+1)-м диапазонами, j = 0..m-2
def fuzzy_entropy_optimal_partition(x, y, z, m, t0, iter_num, lam, simplified=False):
    t = t0.copy()
    J_pred, _ = calc_reduced_correspondence_matrix(x, y, z, t, simplified=simplified)
    t_pred = t.copy()
    cnt = 0
    for it in range(iter_num):
        # TODO: прокомментировать алгоритм
        J_t = calc_derivatives_J_entropy_t(t, x, y, z, simplified=simplified)

        #J_t_check = calc_derivatives_J_t_check(t, x, y)
        #print("J_t =", J_t)
        #print("J_t_check =", J_t_check)
        #for j in range(m - 1):

        j = random.choice(range(m - 1))
        t_j_new = t[j] - lam * J_t[j]
        if j > 0 and t_j_new < (t[j - 1] + t[j]) / 2:
            t_j_new = (t[j - 1] + t[j]) / 2
        elif j < m - 2 and t_j_new > (t[j] + t[j + 1]) / 2:
            t_j_new = (t[j] + t[j + 1]) / 2
        t[j] = t_j_new

        J, _ = calc_reduced_correspondence_matrix(x, y, z, t, simplified=simplified)
        if J < J_pred:
            cnt += 1
            if cnt == 5:
                lam *= 1.3
                cnt = 0
            t_pred = t.copy()
            J_pred = J
        else:
            lam /= 1.5
            cnt = 0
            t = t_pred.copy()
        print("   iteration", it)
        print("   t = ", t)
        print("   J =", J)
        #print("   lam =", lam)
    return t
