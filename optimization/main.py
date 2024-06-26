import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from input_data import *
from optimize_kpi import optimize, calc_kpi


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

z, x_new, q_new = optimize(compet_t0.x[selected, :], expectations.q[selected, :], expectations.a[selected, :],
         invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
         budget_constraints, total_budget)

print("Распределение по направлениям:", np.sum(z, axis=0))

csvfile = open('result.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
for i in range(z.shape[0]):
    csvwriter.writerow([str(z[i, k]) for k in range(z.shape[1])])

csvfile = open('x.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
for i in range(compet_t0.x[selected, :].shape[0]):
    csvwriter.writerow([str(compet_t0.x[selected, :][i, j]) for j in range(compet_t0.x[selected, :].shape[1])])

csvfile = open('x_new.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
for i in range(x_new.shape[0]):
    csvwriter.writerow([str(x_new[i, j]) for j in range(x_new.shape[1])])

csvfile = open('q.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
for i in range(expectations.q[selected, :].shape[0]):
    csvwriter.writerow([str(expectations.q[selected, :][i, k]) for k in range(expectations.q[selected, :].shape[1])])

csvfile = open('q_new.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
for i in range(q_new.shape[0]):
    csvwriter.writerow([str(q_new[i, k]) for k in range(q_new.shape[1])])


kpi1 = calc_kpi(x_new, q_new, expectations.a[selected, :], expectations_to_burnout, compet_burnout_to_kpi)
print("Прогноз KPI: ", np.mean(kpi1, axis=0), " -> ", np.dot(np.mean(kpi1, axis=0), compet_burnout_to_kpi.kpi_importance))
print("Реальные KPI при t = 0: ", np.mean(kpi_t0.y[selected, :], axis=0), " -> ",
      np.dot(np.mean(kpi_t0.y[selected, :], axis=0), compet_burnout_to_kpi.kpi_importance))


plt.scatter(np.dot(kpi_t0.y[selected, :], compet_burnout_to_kpi.kpi_importance),
            np.sum(z, axis=1), c=np.mean(burnout_t0.b[selected, :], axis=1), cmap='Reds')
plt.xlabel("Агрегированный KPI при t = 0")
plt.ylabel("Инвестиции в сотрудника")
plt.show()

plt.scatter(np.dot(compet_t0.x[selected, :], compet_burnout_to_kpi.w[0, :]),
            np.sum(z, axis=1), c=np.mean(burnout_t0.b[selected, :], axis=1), cmap='Reds')
plt.xlabel("Интегральный показатель компетентности при t = 0")
plt.ylabel("Инвестиции в сотрудника")
plt.show()

plt.scatter(np.mean(burnout_t0.b[selected, :], axis=1), np.sum(z, axis=1))
plt.xlabel("Выгорание при t = 0")
plt.ylabel("Инвестиции в сотрудника")
plt.show()

plt.scatter(np.dot(expectations.q[selected, :] * expectations.a[selected, :], expectations_to_burnout.w[0, :]),
            np.sum(z, axis=1), c=np.mean(burnout_t0.b[selected, :], axis=1), cmap='Reds')
plt.xlabel("Интегральный показатель ожиданий при t = 0")
plt.ylabel("Инвестиции в сотрудника")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=123, max_iter=100)
cluster_labels = kmeans.fit_predict(np.abs(expectations.a[selected, :]))
print(kmeans.cluster_centers_)
#plt.scatter(np.dot(expectations.q[selected, :] * expectations.a[selected, :], expectations_to_burnout.w[0, :]),
#            np.sum(z, axis=1), c=kmeans.labels_, marker='o')
plt.scatter(np.dot(expectations.q[selected, :][cluster_labels == 0, :] * expectations.a[selected, :][cluster_labels == 0, :], expectations_to_burnout.w[0, :]),
            np.sum(z[cluster_labels == 0], axis=1), c='red')
plt.scatter(np.dot(expectations.q[selected, :][cluster_labels == 1, :] * expectations.a[selected, :][cluster_labels == 1, :], expectations_to_burnout.w[0, :]),
            np.sum(z[cluster_labels == 1], axis=1), c='blue')
plt.scatter(np.dot(expectations.q[selected, :][cluster_labels == 2, :] * expectations.a[selected, :][cluster_labels == 2, :], expectations_to_burnout.w[0, :]),
            np.sum(z[cluster_labels == 2], axis=1), c='green')
plt.xlabel("Интегральный показатель ожиданий при t = 0")
plt.ylabel("Инвестиции в сотрудника")
plt.show()



# еще 3 квартала
sum_z = z.copy()
z_quarterly_empl = np.sum(z, axis=1)
z_quarterly_activities = np.sum(z, axis=0)
for quart in range(2, 5):
    z, x_new, q_new = optimize(x_new, q_new, expectations.a[selected, :],
             invest_to_compet, activities_expectations, expectations_to_burnout, compet_burnout_to_kpi,
             budget_constraints, total_budget)
    sum_z += z
    z_quarterly_empl = np.c_[z_quarterly_empl, np.sum(z, axis=1)]
    z_quarterly_activities = np.c_[z_quarterly_activities, np.sum(z, axis=0)]
    print("Квартал", quart)
    print("Распределение по направлениям:", np.sum(z, axis=0))
    kpi1 = calc_kpi(x_new, q_new, expectations.a[selected, :], expectations_to_burnout, compet_burnout_to_kpi)
    print("Прогноз KPI: ", np.mean(kpi1, axis=0), " -> ", np.dot(np.mean(kpi1, axis=0), compet_burnout_to_kpi.kpi_importance))

print("z_quarterly_empl =", z_quarterly_empl)
print("z_quarterly_activities =", z_quarterly_activities)

csvfile = open('result_year.csv', 'w', newline='')
csvwriter = csv.writer(csvfile, delimiter=';')
for i in range(sum_z.shape[0]):
    csvwriter.writerow([str(sum_z[i, k]) for k in range(sum_z.shape[1])])

# TODO: вывести таблицу по людям, сколько в них вложили в каждый из кварталов


def plot_expectations_to_burnout(l):
    integral_expectations = np.dot(expectations.q * expectations.a, expectations_to_burnout.w[l, :])
    min_x = np.min(integral_expectations)
    max_x = np.max(integral_expectations)
    y_min = 0
    y_max = 100
    plt.plot(integral_expectations, burnout_t0.b[:, l], 'ro')
    for i in range(1, num_expectation_classes):
        plt.plot([expectations_to_burnout.t[l, i], expectations_to_burnout.t[l, i]], [0, 100], 'g')
    for i in range(num_expectation_classes):
        plt.plot([min_x, max_x], [i * (y_max - y_min) / num_expectation_classes,
                                  i * (y_max - y_min) / num_expectation_classes], 'b')
    avg_x = np.array([(expectations_to_burnout.t[l, i] + expectations_to_burnout.t[l, i + 1]) / 2
                      for i in range(num_expectation_classes)])
    avg_y = np.array([(i + 0.5) * (y_max - y_min) / num_expectation_classes
                      for i in range(num_expectation_classes)])
    avg_z = np.dot(expectations_to_burnout.corr_matrix[l, :, :], avg_y)
    plt.plot(np.hstack([[min_x], avg_x, [max_x]]), np.hstack([[avg_z[0]], avg_z, [avg_z[num_expectation_classes - 1]]]), 'k', linewidth=3)

    plt.xlabel("Интегральный показатель ожиданий")
    plt.ylabel("Показатель выгорания")
    plt.show()

def plot_compet_to_kpi(m):
    integral_compet = np.dot(compet_t0.x, compet_burnout_to_kpi.w[m, :])
    min_x = np.min(integral_compet)
    max_x = np.max(integral_compet)
    min_y = np.min(kpi_t0.y[:, m])
    max_y = np.max(kpi_t0.y[:, m])

    #plt.plot(integral_compet, kpi_t0.y[:, m], 'ro')
    plt.scatter(integral_compet, kpi_t0.y[:, m], c=np.mean(burnout_t0.b, axis=1), cmap='Reds')

    for i in range(1, num_compet_classes):
        plt.plot([compet_burnout_to_kpi.t[m, i], compet_burnout_to_kpi.t[m, i]], [min_y, max_y], 'g')

    for i in range(num_compet_classes):
        x1 = compet_burnout_to_kpi.t[m, i]
        x2 = compet_burnout_to_kpi.t[m, i + 1]
        plt.plot([x1, x2], [compet_burnout_to_kpi.c[m, i], compet_burnout_to_kpi.c[m, i]], 'b', linestyle='dashed')

    avg_x = np.array([(compet_burnout_to_kpi.t[m, i] + compet_burnout_to_kpi.t[m, i + 1]) / 2
                      for i in range(num_compet_classes)])
    avg_y = compet_burnout_to_kpi.c[m, :]
    plt.plot(np.hstack([[min_x], avg_x, [max_x]]), np.hstack([[avg_y[0]], avg_y, [avg_y[num_compet_classes - 1]]]), 'k', linewidth=3)

    plt.xlabel("Интегральный показатель компетентности")
    plt.ylabel("KPI")
    plt.show()


# раскомментировать, если нужны графики
#plot_expectations_to_burnout(0)
#plot_compet_to_kpi(0)
