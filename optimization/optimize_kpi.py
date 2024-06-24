import numpy as np

# Вход:
#   x - массив I x J - компетенции сотрудников в момент времени t
#   q - массив I x K - отклонения в ожиданиях сотрудников от мероприятий в момент времени t
#   a - массив I x K - важность мероприятий для сотрудников
# Выход:
#   z - массив I x K - инвестиции в сотрудников по направлениям
#   x_new - массив I x J - компетенции сотрудников в момент времени t + 1
#   q_new - массив I x K - отклонения в ожиданиях сотрудников от мероприятий в момент времени t + 1
def optimize(x, q, a, invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
             budget_constraints, total_budget):
    data_size = x.shape[0]
    num_compet = x.shape[1]
    num_activities = q.shape[1]
    num_kpi_indicators = compet_burnout_to_kpi.w.shape[0]
    num_compet_classes = compet_burnout_to_kpi.c.shape[1]

    # количество инвестиций в каждого сотрудника по каждому из направлений
    z = np.zeros((data_size, num_activities))

    # ограничения на инвестиции в каждое направление
    budget_activities = budget_constraints.budget_activities_percent / 100 * total_budget

    budget_spent = 0
    x_eps = 1e-9  # точность сравнения показателей компетенций
    z_eps = 10
    x_new = np.zeros((data_size, num_compet))
    q_new = np.zeros((data_size, num_activities))
    # инициализируем компетенции
    for i in range(data_size):
        for j in range(num_compet):
            x_new[i, j] = invest_to_compet.beta[j] * x[i, j]
    while budget_spent < total_budget:
        best_i = None
        best_k = None
        max_coef = None
        increase_z_ik = None
        for i in range(data_size):
            for k in range(num_activities):
                # максимальное приращение z[i, k]
                max_increase_z_ik = activities_expectations.nu[k] - z[i, k]
                max_increase_z_ik = min(max_increase_z_ik, budget_activities[k] - np.sum(z[:, k]))
                max_increase_z_ik = min(max_increase_z_ik, total_budget - np.sum(z[:, :]))
                # Далее мы упрощаем: добираемся сначала до ближайшей точки излома, в действительности нужно
                #   перебрать все последующие точки излома и вычислить каждый раз коэффициент при z_ik
                coef_kpi_z_ik = 0  # коэффициент при z_ik в формуле для интегрального KPI
                for m in range(num_kpi_indicators):
                    # перебираем по очереди все интегральные показатели компетентности
                    # вычисляем текущее значение интегрального показателя
                    integral_compet = np.dot(x_new[i, :], compet_burnout_to_kpi.w[m, :])
                    # найдем ближайшую справа точку излома
                    if integral_compet > compet_burnout_to_kpi.r[m, num_compet_classes - 1] - x_eps:
                        # если текущая точка правее крайней точки излома, пропускаем
                        continue
                    p = 0
                    while integral_compet > compet_burnout_to_kpi.r[m, p] - x_eps:
                        p += 1
                    # p - номер ближайшей справа точки излома
                    # вычислим коэффициент зависимости m-го интегрального показателя компетентности от z_ik
                    coef_integral_z_ik = np.dot(compet_burnout_to_kpi.w[m, :], invest_to_compet.alpha[:, k])
                    # вычислим, на сколько мы можем повысить m-й интегральный показатель компетентности
                    dist = compet_burnout_to_kpi.r[m, p] - integral_compet
                    # вычислим угловой коэффициент влияния m-го интегрального показателя компетентности на m-й KPI
                    if p == 0:
                        slope = 0
                    else:
                        slope = (compet_burnout_to_kpi.c[m, p] - compet_burnout_to_kpi.c[m, p - 1]) /\
                                (compet_burnout_to_kpi.r[m, p] - compet_burnout_to_kpi.r[m, p - 1])
                    # рассчитываем коэффициент при z_ik в формуле для KPI
                    coef_kpi_z_ik += compet_burnout_to_kpi.kpi_importance[m] * coef_integral_z_ik * slope
                    if coef_integral_z_ik < 0:
                        continue
                    # вычислим, на сколько мы можем повысить z_ik
                    max_increase_z_ik = min(max_increase_z_ik, dist / coef_integral_z_ik)

                # добавим коэффициент влияния на ожидания => выгорание
                for m in range(num_kpi_indicators):
                    # Также здесь мы упрощаем зависимость выгорания от ожиданий: эту зависимость мы линеаризуем
                    # использовать переменную q_new
                    pass

              #  print(i, k, max_increase_z_ik, coef_kpi_z_ik)

                if (max_coef is None or coef_kpi_z_ik > max_coef) and max_increase_z_ik > z_eps:
                    max_coef = coef_kpi_z_ik
                    best_i = i
                    best_k = k
                    increase_z_ik = max_increase_z_ik

        # вкладываем инвестиции в выбранного i-го человека и выбранное k-е мероприятие
        z[best_i, best_k] += increase_z_ik
        budget_spent += increase_z_ik
        for j in range(num_compet):
            x_new[best_i, j] += invest_to_compet.alpha[j, best_k] * increase_z_ik
        # TODO: записать, чему будет равно q_new


                # не забыть придать приращение x_new, q_new, z и budget_spent
        #break
        print("max_coef =", max_coef)
        print("increase_z_ik =", increase_z_ik)
        print("Потрачено:", budget_spent)

    return z, x_new, q_new


# Расчет KPI в текущий момент времени для всех сотрудников
# Вход:
#   x - массив I x J - компетенции сотрудников в текущий момент времени
#   q - массив I x K - отклонения в ожиданиях сотрудников от мероприятий в текущий момент времени
#   a - массив I x K - важность мероприятий для сотрудников
def calc_kpi(x, q, a, expectations_to_burnout, compet_burnout_to_kpi):
    data_size = x.shape[0]
    num_kpi_indicators = compet_burnout_to_kpi.w.shape[0]
    num_burnout_indicators = compet_burnout_to_kpi.burnout_coef.shape[1]
    kpi = np.zeros((data_size, num_kpi_indicators))

    for m in range(num_kpi_indicators):
        # интегральный показатель компетентности для каждого сотрудника
        #integral_compet = np.dot(x, compet_burnout_to_kpi.w[m, :])
        for i in range(data_size):
            # считаем выгорание
            b = np.zeros(num_burnout_indicators)
            for l in range(num_burnout_indicators):
                b[l] = expectations_to_burnout.calc_burnout(l, a[i, :], q[i, :])
            # считаем KPI
            kpi[i, m] = compet_burnout_to_kpi.calc_kpi(m, x[i, :], b)

    return kpi
