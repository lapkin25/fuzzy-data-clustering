import matplotlib.pyplot as plt
from input_data import *


invest_to_compet = InvestToCompet("invest_to_compet.csv")
compet_t0 = CompetData("data_compet_t0.csv")
expectations = ExpectationsData("data_deviation_expectations.csv")
burnout_t0 = BurnoutData("data_burnout_t0.csv")
kpi_t0 = KPIData("data_kpi_t0.csv")

#activities_expectations = ActivitiesExpectations("expectations.csv")
expectations_to_burnout = ExpectationsToBurnout(expectations)
compet_burnout_to_kpi = CompetBurnoutToKPI(compet_t0)


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


plot_expectations_to_burnout(0)
plot_compet_to_kpi(0)
