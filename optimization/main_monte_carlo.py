from input_data import *
from optimize_kpi import optimize, calc_kpi
import random


# исходные данные
expectations = ExpectationsData("data_deviation_expectations.csv")
compet_t0 = CompetData("data_compet_t0.csv")
burnout_t0 = BurnoutData("data_burnout_t0.csv")
kpi_t0 = KPIData("data_kpi_t0.csv")

# эконометрические зависимости
invest_to_compet = InvestToCompet("invest_to_compet.csv")
activities_expectations = ActivitiesExpectations("data_min_max_expectations.csv")
expectations_to_burnout = ExpectationsToBurnout(expectations)
compet_burnout_to_kpi = CompetBurnoutToKPI(compet_t0)

# ограничения оптимизации
budget_constraints = BudgetConstraints("budget_activities.csv")

# создаем случайную выборку из 100 людей
selected_data_size = 100
random_state = 1
np.random.seed(random_state)
indices = np.random.permutation(data_size)
selected = indices[:selected_data_size]
print("Индексы выбранных сотрудников: ", selected)

# бюджет: 500000 в год на человека
total_budget = 500000 / 4 * selected_data_size


# [a, b] - это доверительный интервал mu ± 2 * sigma
def trunc_normal_random(a, b):
    ok = False
    while not ok:
        x = random.gauss(mu=(a + b) / 2, sigma=(b - a) / 4)
        if x >= a and x <= b:
            ok = True
    #print(a, b, x)
    return x


# Варьируем коэффициенты модели, разыгрывая их на доверительных интервалах
# Находим случайную реализацию оптимального KPI
def generate_random_kpi():
    # Разыгрываем коэффициенты влияния инвестиций на компетенции
    perturbed_invest_to_compet = InvestToCompet("invest_to_compet.csv")
    # просто присвоить один объект другому нельзя, поэтому создали еще один такой же
    for j in range(num_compet):
        #print("Проверка:", invest_to_compet.beta[j], "[", invest_to_compet_left_conf.beta[j], ",",
        #      invest_to_compet_right_conf.beta[j], "]")
        for k in range(num_activities):
            #print("Проверка:", invest_to_compet.alpha[j, k], "[", invest_to_compet_left_conf.alpha[j, k], ",",
            #      invest_to_compet_right_conf.alpha[j, k], "]")
            perturbed_invest_to_compet.beta[j] =\
                trunc_normal_random(invest_to_compet_left_conf.beta[j], invest_to_compet_right_conf.beta[j])
            perturbed_invest_to_compet.alpha[j, k] =\
                trunc_normal_random(invest_to_compet_left_conf.alpha[j, k], invest_to_compet_right_conf.alpha[j, k])

    z, x_new, q_new = optimize(compet_t0.x[selected, :], expectations.q[selected, :], expectations.a[selected, :],
                               perturbed_invest_to_compet, activities_expectations, expectations_to_burnout,
                               compet_burnout_to_kpi, budget_constraints, total_budget)

    kpi1 = calc_kpi(x_new, q_new, expectations.a[selected, :], expectations_to_burnout, compet_burnout_to_kpi)
    integral_kpi = np.dot(np.mean(kpi1, axis=0), compet_burnout_to_kpi.kpi_importance)

    return integral_kpi


print("Реальные KPI при t = 0: ", np.mean(kpi_t0.y[selected, :], axis=0), " -> ",
    np.dot(np.mean(kpi_t0.y[selected, :], axis=0), compet_burnout_to_kpi.kpi_importance))

invest_to_compet_left_conf = InvestToCompet("invest_to_compet_left_conf_interval.csv")
invest_to_compet_right_conf = InvestToCompet("invest_to_compet_right_conf_interval.csv")

num_samples = 10
with open('kpi_realizations.txt', 'w') as fout_kpi:
    for _ in range(num_samples):
        kpi_realization = generate_random_kpi()
        print(kpi_realization, file=fout_kpi)
