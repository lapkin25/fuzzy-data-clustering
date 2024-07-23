import matplotlib.pyplot as plt
from input_data import *


# исходные данные
expectations = ExpectationsData("data_deviation_expectations.csv")
burnout_t0 = BurnoutData("data_burnout_t0.csv")
# TODO: исходные данные по выгоранию должны быть сразу агрегированными

# эконометрические зависимости
expectations_to_burnout = ExpectationsToBurnout(expectations)


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


plot_expectations_to_burnout()
