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
    z = np.zeros((data_size, num_activities))

    # ограничения на инвестиции в каждое направление
    budget_activities = budget_constraints.budget_activities_percent / 100 * total_budget

    x_new = np.zeros((data_size, num_compet))
    q_new = np.zeros((data_size, num_activities))

    return z, x_new, q_new


# Расчет KPI в текущий момент времени для всех сотрудников
# Вход:
#   x - массив I x J - компетенции сотрудников в текущий момент времени
#   q - массив I x K - отклонения в ожиданиях сотрудников от мероприятий в текущий момент времени
#   a - массив I x K - важность мероприятий для сотрудников
def calc_kpi(x, q, a, expectations_to_burnout, compet_burnout_to_kpi):
    data_size = x.shape[0]
    num_kpi_indicators = compet_burnout_to_kpi.w.shape[0]
    kpi = np.zeros((data_size, num_kpi_indicators))

    for m in range(num_kpi_indicators):
        # интегральный показатель компетентности для каждого сотрудника
        integral_compet = np.dot(x, compet_burnout_to_kpi.w[m, :])

    return kpi
