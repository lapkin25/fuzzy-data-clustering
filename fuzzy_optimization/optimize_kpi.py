import numpy as np


# Вход:
#   x - массив I x J - компетенции сотрудников в момент времени t
#   q - массив I x K - отклонения в ожиданиях сотрудников от мероприятий в момент времени t
#   a - массив I x K - важность мероприятий для сотрудников
# Выход:
#   z - массив I x K - инвестиции в сотрудников по направлениям
def optimize1(x, q, a, invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
              budget_constraints, total_budget, budget1):
    data_size = x.shape[0]
    num_compet = x.shape[1]
    num_activities = q.shape[1]
    num_compet_classes = compet_burnout_to_kpi.c.shape[0]
    num_expectation_classes = expectations_to_burnout.r.shape[0]

    # количество инвестиций в каждого сотрудника по каждому из направлений
    z = np.zeros((data_size, num_activities), dtype=float)

    # ограничения на инвестиции в каждое направление
    budget_activities = budget_constraints.budget_activities_percent / 100 * total_budget

    # коэффициент наклона зависимости KPI от интегрального показателя компетентности для каждого сотрудника
    delta = np.zeros(data_size)
    # Далее вычисляем delta[i] для всех i
    # сначала находим интегральный показатель компетентности
    integral_compet = np.dot(x, compet_burnout_to_kpi.w)
    # затем вычисляем коэффициент наклона на границах диапазонов интегрального показателя компетентности
    range_slopes = np.zeros(num_compet_classes + 1)
    for p in range(1, num_compet_classes):
        range_slopes[p] = (compet_burnout_to_kpi.c[p] - compet_burnout_to_kpi.c[p - 1]) \
                          / ((compet_burnout_to_kpi.t[p + 1] - compet_burnout_to_kpi.t[p - 1]) / 2)
    # теперь находим коэффициенты наклона, интерполируя range_slopes[p]
    for i in range(data_size):
        if integral_compet[i] <= compet_burnout_to_kpi.t[0]:
            delta[i] = 0.0
        elif integral_compet[i] >= compet_burnout_to_kpi.t[num_compet_classes]:
            delta[i] = 0.0
        else:
            p = 0
            while compet_burnout_to_kpi.t[p] < integral_compet[i]:
                p += 1
            # p равно наименьшему индексу, такому, что t[p] >= integral_compet[i]
            # p >= 1
            lam = (integral_compet[i] - compet_burnout_to_kpi.t[p - 1])\
                  / (compet_burnout_to_kpi.t[p] - compet_burnout_to_kpi.t[p - 1])
            delta[i] = (1 - lam) * range_slopes[p - 1] + lam * range_slopes[p]

    # print(delta)



    #print(budget_activities)

    #x_new = np.zeros((data_size, num_compet))
    # инициализируем компетенции
#    for i in range(data_size):
#        for j in range(num_compet):
#            x_new[i, j] = invest_to_compet.beta[j] * x[i, j]


    return z


# Расчет KPI в текущий момент времени для всех сотрудников
# Вход:
#   x - массив I x J - компетенции сотрудников в текущий момент времени
#   q - массив I x K - отклонения в ожиданиях сотрудников от мероприятий в текущий момент времени
#   a - массив I x K - важность мероприятий для сотрудников
def calc_kpi(x, q, a, expectations_to_burnout, compet_burnout_to_kpi):
    data_size = x.shape[0]
    kpi = np.zeros(data_size)

    # интегральный показатель компетентности для каждого сотрудника
    #integral_compet = np.dot(x, compet_burnout_to_kpi.w[m, :])
    for i in range(data_size):
        # считаем выгорание
        b = expectations_to_burnout.calc_burnout(a[i, :], q[i, :])
        # считаем KPI
        kpi[i] = compet_burnout_to_kpi.calc_kpi(x[i, :], b)

    return kpi
