import matplotlib.pyplot as plt
from input_data import *
from optimize_kpi import optimize1, calc_kpi


# исходные данные
expectations = ExpectationsData("data_deviation_expectations.csv")
compet_t0 = CompetData("data_compet_t0.csv")
burnout_t0 = BurnoutData("data_burnout_t0.csv")
kpi_t0 = KPIData("data_kpi_t0.csv")
# TODO: исходные данные по выгоранию должны быть сразу агрегированными

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

budget1 = total_budget / 2

# здесь потом будет вызов общей функции optimize с тем же интерфейсом, что и раньше
#   с последующим расчетом KPI
z = optimize1(compet_t0.x[selected, :], expectations.q[selected, :], expectations.a[selected, :],
              invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
              budget_constraints, total_budget, budget1)


def plot_expectations_to_burnout():
    integral_expectations = np.dot(expectations.q * expectations.a, expectations_to_burnout.w)
    min_x = np.min(integral_expectations)
    max_x = np.max(integral_expectations)
    y_min = 0
    y_max = 100
    plt.plot(integral_expectations, burnout_t0.b, 'ro')
    for i in range(1, num_expectation_classes):
        plt.plot([expectations_to_burnout.t[i], expectations_to_burnout.t[i]], [0, 100], 'g')
    for i in range(num_expectation_classes):
        plt.plot([min_x, max_x], [i * (y_max - y_min) / num_expectation_classes,
                                  i * (y_max - y_min) / num_expectation_classes], 'b')
    avg_x = np.array([(expectations_to_burnout.t[i] + expectations_to_burnout.t[i + 1]) / 2
                      for i in range(num_expectation_classes)])
    avg_y = np.array([(i + 0.5) * (y_max - y_min) / num_expectation_classes
                      for i in range(num_expectation_classes)])
    avg_z = np.dot(expectations_to_burnout.corr_matrix, avg_y)
    plt.plot(np.hstack([[min_x], avg_x, [max_x]]), np.hstack([[avg_z[0]], avg_z, [avg_z[num_expectation_classes - 1]]]), 'k', linewidth=3)

    plt.xlabel("Интегральный показатель ожиданий")
    plt.ylabel("Показатель выгорания")
    plt.show()

def plot_compet_to_kpi():
    integral_compet = np.dot(compet_t0.x, compet_burnout_to_kpi.w)
    min_x = np.min(integral_compet)
    max_x = np.max(integral_compet)
    integral_kpi = np.dot(kpi_t0.y, compet_burnout_to_kpi.kpi_importance)
    min_y = np.min(integral_kpi)
    max_y = np.max(integral_kpi)

    #plt.plot(integral_compet, kpi_t0.y[:, m], 'ro')
    plt.scatter(integral_compet, integral_kpi, c=burnout_t0.b, cmap='Reds')

    for i in range(1, num_compet_classes):
        plt.plot([compet_burnout_to_kpi.t[i], compet_burnout_to_kpi.t[i]], [min_y, max_y], 'g')

    for i in range(num_compet_classes):
        x1 = compet_burnout_to_kpi.t[i]
        x2 = compet_burnout_to_kpi.t[i + 1]
        plt.plot([x1, x2], [compet_burnout_to_kpi.c[i], compet_burnout_to_kpi.c[i]], 'b', linestyle='dashed')

    avg_x = np.array([(compet_burnout_to_kpi.t[i] + compet_burnout_to_kpi.t[i + 1]) / 2
                      for i in range(num_compet_classes)])
    avg_y = compet_burnout_to_kpi.c
    plt.plot(np.hstack([[min_x], avg_x, [max_x]]), np.hstack([[avg_y[0]], avg_y, [avg_y[num_compet_classes - 1]]]), 'k', linewidth=3)

    plt.xlabel("Интегральный показатель компетентности")
    plt.ylabel("KPI")
    plt.show()


#plot_expectations_to_burnout()
#plot_compet_to_kpi()