import numpy as np
from scipy import optimize
from scipy.optimize import linprog, milp


def get_person_var_index(cost_spent, k, cost_index):
    return sum([len(cost_spent[j]) for j in range(k)]) + cost_index


def get_person_var_params(cost_spent, var_index):
    s = 0
    k = 0
    while var_index >= s + len(cost_spent[k]):
        s += len(cost_spent[k])
        k += 1
    return k, var_index - s


def get_var_index(cost_spent, i, k, cost_index, num_vars_person, data_size):
    return i * num_vars_person + get_person_var_index(cost_spent, k, cost_index)


def get_var_params(cost_spent, var_index, num_vars_person, data_size):
    k, cost_index = get_person_var_params(cost_spent, var_index % num_vars_person)
    i = var_index // num_vars_person
    return i, k, cost_index


def optimize_full(x, q, a, invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
              budget_constraints, total_budget, activities, compet_growth):
    # TODO: Свести задачу к задаче целочисленного линейного программирования
    # Не забыть про единицы измерения (тыс. руб.)

    data_size = x.shape[0]
    num_activities = q.shape[1]
    num_compet = x.shape[1]
    num_compet_classes = compet_burnout_to_kpi.c.shape[0]
    num_expectation_classes = expectations_to_burnout.r.shape[0]

    # количество инвестиций (единицы измерения - рубли) в каждого сотрудника по каждому из направлений
    z = np.zeros((data_size, num_activities), dtype=int)

    # ограничения на инвестиции в каждое направление
    budget_activities = budget_constraints.budget_activities_percent / 100 * total_budget

    # Найдем, сколько денег можно потратить на каждое направление мероприятий
    cost_spent = [np.array([0], dtype=int) for _ in range(num_activities)]
    # cost_spent - список возможных сумм, которые можно потратить на каждое направление мероприятий
    for k in range(num_activities):
        sum_cost = sum([s['cost'] // 1000 for s in activities.activities_lists[k]])
        can_spend = np.zeros(sum_cost + 1, dtype=bool)
        can_spend[0] = True
        for s in activities.activities_lists[k]:
            pred_can_spend = can_spend.copy()
            for c in range(0, sum_cost + 1):
                # c - сколько потратили на просмотренные мероприятия (тыс. руб.)
                if pred_can_spend[c]:
                    cost = s['cost'] // 1000
                    # cost - сколько стоит очередное мероприятие (тыс. руб.)
                    can_spend[c + cost] = True
        for c in range(1, sum_cost + 1):
            if can_spend[c]:
                cost_spent[k] = np.append(cost_spent[k], c)

    for k in range(num_activities):
        print(cost_spent[k])

    # количество переменных на одного человека
    num_vars_person = sum([len(cost_spent[k]) for k in range(num_activities)])
    num_vars = data_size * num_vars_person
    bounds = optimize.Bounds(0, 1)

    """
    for i in range(data_size):
        for k in range(num_activities):
            for j, cost in enumerate(cost_spent[k]):
                var_index = get_var_index(cost_spent, i, k, j, num_vars_person, data_size)
                i1, k1, j1 = get_var_params(cost_spent, var_index, num_vars_person, data_size)
                print(i, k, j, i1, k1, j1)
    """

    # коэффициенты при переменных в ограничениях
    # data_size * num_activities ограничений: для каждого человека и каждого мероприятия
    # ограничение: среди всех коэффициентов для отдельного человека и отдельного мероприятия должна быть ровно одна 1
    param_single_constr = [np.zeros((num_activities, num_vars)) for _ in range(data_size)]
    param_single_constr_rhs_l = [np.zeros(num_activities) for _ in range(data_size)]
    param_single_constr_rhs_u = [np.zeros(num_activities) for _ in range(data_size)]
    for i in range(data_size):
        for k in range(num_activities):
            for j, cost in enumerate(cost_spent[k]):
                var_index = get_var_index(cost_spent, i, k, j, num_vars_person, data_size)
                param_single_constr[i][k, var_index] = 1
            param_single_constr_rhs_l[i][k] = 1
            param_single_constr_rhs_u[i][k] = 1

    # следующая группа ограничений: сумма вложений в людей в рамках определенного мероприятия
    param_group_constr = np.zeros((num_activities, num_vars))
    param_group_constr_rhs_u = np.zeros(num_activities)
    param_group_constr_rhs_l = np.zeros(num_activities)
    for k in range(num_activities):
        for i in range(data_size):
            for j, cost in enumerate(cost_spent[k]):
                var_index = get_var_index(cost_spent, i, k, j, num_vars_person, data_size)
                param_group_constr[k, var_index] = cost * 1000
    for k in range(num_activities):
        param_group_constr_rhs_u[k] = budget_activities[k]

    # суммарное ограничение
    param_sum_constr = np.zeros(num_vars)
    for k in range(num_activities):
        for i in range(data_size):
            for j, cost in enumerate(cost_spent[k]):
                var_index = get_var_index(cost_spent, i, k, j, num_vars_person, data_size)
                param_sum_constr[var_index] = cost * 1000
    param_sum_constr_rhs_l = 0
    param_sum_constr_rhs_u = total_budget

    # коэффициенты при переменных в линейных ограничениях
    param_lin_constr = np.zeros((data_size, num_vars))
    for i in range(data_size):
        for k in range(num_activities):
            for j, cost in enumerate(cost_spent[k]):
                var_index = get_var_index(cost_spent, i, k, j, num_vars_person, data_size)
                param_lin_constr[i, var_index] = np.dot(invest_to_compet.alpha[:, k], compet_burnout_to_kpi.w)\
                                                 * cost * 1000
    param_lin_constr_rhs_u = np.ones(data_size) * compet_growth
    param_lin_constr_rhs_l = np.zeros(data_size)

    all_constr = np.vstack(param_single_constr)
    all_constr = np.vstack([all_constr, param_group_constr, param_lin_constr, [param_sum_constr]])
    #all_constr = np.vstack([all_constr, param_group_constr, [param_sum_constr]])
    all_constr_rhs_u = np.hstack(param_single_constr_rhs_u)
    all_constr_rhs_u = np.hstack([all_constr_rhs_u, param_group_constr_rhs_u, param_lin_constr_rhs_u, [param_sum_constr_rhs_u]])
    #all_constr_rhs_u = np.hstack([all_constr_rhs_u, param_group_constr_rhs_u, [param_sum_constr_rhs_u]])
    all_constr_rhs_l = np.hstack(param_single_constr_rhs_l)
    all_constr_rhs_l = np.hstack([all_constr_rhs_l, param_group_constr_rhs_l, param_lin_constr_rhs_l, [param_sum_constr_rhs_l]])
    #all_constr_rhs_l = np.hstack([all_constr_rhs_l, param_group_constr_rhs_l, [param_sum_constr_rhs_l]])

    constraints = optimize.LinearConstraint(A=all_constr, lb=all_constr_rhs_l, ub=all_constr_rhs_u)

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

    # коэффициент наклона зависимости выгорания от интегрального показателя ожиданий для каждого сотрудника
    epsilon = np.zeros(data_size)
    # сначала находим интегральный показатель ожиданий
    integral_expectations = np.dot(q * a, expectations_to_burnout.w)
    # затем вычисляем коэффициент наклона на границах диапазонов интегрального показателя компетентности
    range_slopes = np.zeros(num_expectation_classes + 1)
    for p in range(1, num_expectation_classes):
        range_slopes[p] = (expectations_to_burnout.e[p] - expectations_to_burnout.e[p - 1]) \
                          / ((expectations_to_burnout.t[p + 1] - expectations_to_burnout.t[p - 1]) / 2)
    # теперь находим коэффициенты наклона, интерполируя range_slopes[p]
    for i in range(data_size):
        if integral_expectations[i] <= expectations_to_burnout.t[0]:
            epsilon[i] = 0.0
        elif integral_expectations[i] >= expectations_to_burnout.t[num_expectation_classes]:
            epsilon[i] = 0.0
        else:
            p = 0
            while expectations_to_burnout.t[p] < integral_expectations[i]:
                p += 1
            # p равно наименьшему индексу, такому, что t[p] >= integral_expectations[i]
            # p >= 1
            lam = (integral_expectations[i] - expectations_to_burnout.t[p - 1])\
                  / (expectations_to_burnout.t[p] - expectations_to_burnout.t[p - 1])
            epsilon[i] = (1 - lam) * range_slopes[p - 1] + lam * range_slopes[p]

    obj_coef = np.zeros(num_vars)
    for k in range(num_activities):
        for i in range(data_size):
            for j, cost in enumerate(cost_spent[k]):
                var_index = get_var_index(cost_spent, i, k, j, num_vars_person, data_size)
                compet_coef = cost * 1000 * delta[i] * np.dot(invest_to_compet.alpha[:, k], compet_burnout_to_kpi.w)
                new_expect = activities_expectations.calc_expectations(k, cost * 1000, q[i, k])
                expect_coef = a[i, k] * new_expect * expectations_to_burnout.w[k] * epsilon[i]\
                              * compet_burnout_to_kpi.burnout_coef
                obj_coef[var_index] = compet_coef + expect_coef


    integrality = np.full_like(obj_coef, True)

    res = milp(c=-obj_coef, constraints=constraints, integrality=integrality, bounds=bounds,
               options = {"disp": True, "mip_rel_gap": 0.05})
    #print(res.x)
    for k in range(num_activities):
        for i in range(data_size):
            for j, cost in enumerate(cost_spent[k]):
                var_index = get_var_index(cost_spent, i, k, j, num_vars_person, data_size)
                z[i, k] += cost * 1000 * res.x[var_index]

    x_new = np.zeros((data_size, num_compet))
    for i in range(data_size):
        for j in range(num_compet):
            x_new[i, j] = invest_to_compet.beta[j] * x[i, j]
            for k in range(num_activities):
                x_new[i, j] += invest_to_compet.alpha[j, k] * z[i, k]

    q_new = np.zeros((data_size, num_activities))
    for i in range(data_size):
        for k in range(num_activities):
            q_new[i, k] = activities_expectations.calc_expectations(k, z[i, k], q[i, k])

    return z, x_new, q_new


# Вход:
#   x - массив I x J - компетенции сотрудников в момент времени t
#   q - массив I x K - отклонения в ожиданиях сотрудников от мероприятий в момент времени t
#   a - массив I x K - важность мероприятий для сотрудников
# Выход:
#   z - массив I x K - инвестиции в сотрудников по направлениям
def optimize1(x, q, a, invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
              budget_constraints, total_budget, budget1, compet_growth):
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

    # Решаем задачу линейного программирования

    # коэффициенты при переменных в целевой функции
    obj_coef = np.zeros((data_size, num_activities))
    for i in range(data_size):
        for k in range(num_activities):
            obj_coef[i, k] = delta[i] * np.dot(invest_to_compet.alpha[:, k], compet_burnout_to_kpi.w)
    param_obj = obj_coef.flatten()

    # коэффициенты при переменных в линейных ограничениях
    param_lin_constr = np.zeros((data_size, data_size * num_activities))
    for i in range(data_size):
        lin_constr_coef = np.zeros((data_size, num_activities))
        for k in range(num_activities):
            lin_constr_coef[i, k] = np.dot(invest_to_compet.alpha[:, k], compet_burnout_to_kpi.w)
        param_lin_constr[i, :] = lin_constr_coef.flatten()
    # правая часть линейного ограничения
    param_lin_constr_rhs = np.ones(data_size) * compet_growth

    # коэффициенты при переменных в ограничениях на инвестиции в отдельные направления
    param_simple_constr = np.zeros((num_activities, data_size * num_activities))
    for k in range(num_activities):
        simple_constr_coef = np.zeros((data_size, num_activities))
        simple_constr_coef[:, k] = np.ones(data_size)
        param_simple_constr[k, :] = simple_constr_coef.flatten()
    # правая часть ограничения
    param_simple_constr_rhs = np.zeros(num_activities)
    for k in range(num_activities):
        param_simple_constr_rhs[k] = budget_activities[k]

    # коэффициенты при переменных в суммарном ограничении
    param_sum_constr = np.ones(data_size * num_activities)
    # правая часть ограничения
    param_sum_constr_rhs = budget1

    A_ub = np.vstack([param_lin_constr, param_simple_constr, [param_sum_constr]])
    b_ub = np.hstack([param_lin_constr_rhs, param_simple_constr_rhs, [param_sum_constr_rhs]])

    res = linprog(-param_obj, A_ub=A_ub, b_ub=b_ub)
    #print(res.x)
    z = res.x.reshape((data_size, num_activities))
    print(res.message)

    return z


def optimize2(x, q, a, invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
              budget_constraints, total_budget, budget2, activities, budget1_distr):
    data_size = x.shape[0]
    num_activities = q.shape[1]
    num_expectation_classes = expectations_to_burnout.r.shape[0]

    # Найдем, сколько денег можно потратить на каждое направление мероприятий
    cost_spent = [np.array([], dtype=int) for _ in range(num_activities)]
    # cost_spent - список возможных сумм, которые можно потратить на каждое направление мероприятий
    for k in range(num_activities):
        sum_cost = sum([s['cost'] // 1000 for s in activities.activities_lists[k]])
        can_spend = np.zeros(sum_cost + 1, dtype=bool)
        can_spend[0] = True
        for s in activities.activities_lists[k]:
            pred_can_spend = can_spend.copy()
            for c in range(0, sum_cost + 1):
                # c - сколько потратили на просмотренные мероприятия (тыс. руб.)
                if pred_can_spend[c]:
                    cost = s['cost'] // 1000
                    # cost - сколько стоит очередное мероприятие (тыс. руб.)
                    can_spend[c + cost] = True
        for c in range(1, sum_cost + 1):
            if can_spend[c]:
                cost_spent[k] = np.append(cost_spent[k], c)

    for k in range(num_activities):
        print(cost_spent[k])

    # коэффициент наклона зависимости выгорания от интегрального показателя ожиданий для каждого сотрудника
    delta = np.zeros(data_size)
    # сначала находим интегральный показатель ожиданий
    integral_expectations = np.dot(q * a, expectations_to_burnout.w)
    # затем вычисляем коэффициент наклона на границах диапазонов интегрального показателя компетентности
    range_slopes = np.zeros(num_expectation_classes + 1)
    for p in range(1, num_expectation_classes):
        range_slopes[p] = (expectations_to_burnout.e[p] - expectations_to_burnout.e[p - 1]) \
                          / ((expectations_to_burnout.t[p + 1] - expectations_to_burnout.t[p - 1]) / 2)
    # теперь находим коэффициенты наклона, интерполируя range_slopes[p]
    for i in range(data_size):
        if integral_expectations[i] <= expectations_to_burnout.t[0]:
            delta[i] = 0.0
        elif integral_expectations[i] >= expectations_to_burnout.t[num_expectation_classes]:
            delta[i] = 0.0
        else:
            p = 0
            while expectations_to_burnout.t[p] < integral_expectations[i]:
                p += 1
            # p равно наименьшему индексу, такому, что t[p] >= integral_expectations[i]
            # p >= 1
            lam = (integral_expectations[i] - expectations_to_burnout.t[p - 1])\
                  / (expectations_to_burnout.t[p] - expectations_to_burnout.t[p - 1])
            delta[i] = (1 - lam) * range_slopes[p - 1] + lam * range_slopes[p]

    # TODO: метод динамического программирования



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
