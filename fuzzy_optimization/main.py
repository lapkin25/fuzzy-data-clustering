import matplotlib.pyplot as plt
from input_data import *
from optimize_kpi import optimize_full, calc_kpi


# исходные данные
expectations = ExpectationsData("data_deviation_expectations.csv")
compet_t0 = CompetData("data_compet_t0.csv")
burnout_t0 = BurnoutData("data_burnout_t0.csv")
kpi_t0 = KPIData("data_kpi_t0.csv")
activities = Activities("data_activities.csv")

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

#budget1 = total_budget / 2

compet_growth_year = 20  #10


"""
# Первый этап оптимизации: целевая функция - компетенции
z1 = optimize1(compet_t0.x[selected, :], expectations.q[selected, :], expectations.a[selected, :],
              invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
              budget_constraints, total_budget, budget1, compet_growth_year / 4)
budget1_spent = np.sum(z1)
budget1_distr = np.sum(z1, axis=0)

print("Выделено: ", budget1)
print("Потрачено: ", budget1_spent)
print("Структура инвестиций по направлениям: ", budget1_distr)

csvfile = open('result1.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
for i in range(z1.shape[0]):
    csvwriter.writerow([str(z1[i, k]) for k in range(z1.shape[1])])

# Второй этап оптимизации: целевая функция - выгорание

budget2 = total_budget - budget1_spent

_ = optimize2(compet_t0.x[selected, :], expectations.q[selected, :], expectations.a[selected, :],
              invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
              budget_constraints, total_budget, budget2, activities, budget1_distr)
"""

z, x_new, q_new = optimize_full(compet_t0.x[selected, :], expectations.q[selected, :], expectations.a[selected, :],
                                invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
                                budget_constraints, total_budget, activities, compet_growth_year / 4)

budget_spent = np.sum(z)
budget_distr = np.sum(z, axis=0)

print("Выделено: ", total_budget)
print("Потрачено: ", budget_spent)
print("Структура инвестиций по направлениям: ", budget_distr)

csvfile = open('result.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
for i in range(z.shape[0]):
    csvwriter.writerow([str(z[i, k]) for k in range(z.shape[1])])

kpi1 = calc_kpi(x_new, q_new, expectations.a[selected, :], expectations_to_burnout, compet_burnout_to_kpi)
print("Прогноз KPI: ", np.mean(kpi1))
print("Реальные KPI при t = 0: ", np.mean(kpi_t0.y[selected, :], axis=0), " -> ",
      np.dot(np.mean(kpi_t0.y[selected, :], axis=0), compet_burnout_to_kpi.kpi_importance))


plt.scatter(np.dot(compet_t0.x[selected, :], compet_burnout_to_kpi.w),
            np.sum(z, axis=1), c=burnout_t0.b[selected], cmap='Reds')
plt.xlabel("Интегральный показатель компетентности при t = 0")
plt.ylabel("Инвестиции в сотрудника, руб.")
plt.savefig('fig_1.png', dpi=300)
plt.show()

plt.scatter(burnout_t0.b[selected], np.sum(z, axis=1))
plt.xlabel("Выгорание при t = 0")
plt.ylabel("Инвестиции в сотрудника, руб.")
plt.savefig('fig_2.png', dpi=300)
plt.show()

plt.scatter(np.dot(expectations.q[selected, :] * expectations.a[selected, :], expectations_to_burnout.w),
            np.sum(z, axis=1), c=burnout_t0.b[selected], cmap='Reds')
plt.xlabel("Интегральный показатель ожиданий при t = 0")
plt.ylabel("Инвестиции в сотрудника, руб.")
plt.savefig('fig_3.png', dpi=300)
plt.show()


# еще 3 квартала
sum_z = z.copy()
z_quarterly_empl = np.sum(z, axis=1)
z_quarterly_activities = np.sum(z, axis=0)
x_quarterly = [x_new]
q_quarterly = [q_new]
for quart in range(2, 5):
    z, x_new, q_new = optimize_full(x_new, q_new, expectations.a[selected, :],
                                    invest_to_compet, activities_expectations, expectations_to_burnout,
                                    compet_burnout_to_kpi,
                                    budget_constraints, total_budget, activities, compet_growth_year / 4)

    sum_z += z
    z_quarterly_empl = np.c_[z_quarterly_empl, np.sum(z, axis=1)]
    z_quarterly_activities = np.c_[z_quarterly_activities, np.sum(z, axis=0)]
    x_quarterly.append(x_new)
    q_quarterly.append(q_new)
    print("Квартал", quart)
    print("Распределение по направлениям:", np.sum(z, axis=0))
    kpi1 = calc_kpi(x_new, q_new, expectations.a[selected, :], expectations_to_burnout, compet_burnout_to_kpi)
    print("Прогноз KPI: ", np.mean(kpi1))

print("z_quarterly_empl =", z_quarterly_empl)
print("z_quarterly_activities =", z_quarterly_activities)

csvfile = open('result_year.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
for i in range(sum_z.shape[0]):
    csvwriter.writerow([str(sum_z[i, k]) for k in range(sum_z.shape[1])])


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