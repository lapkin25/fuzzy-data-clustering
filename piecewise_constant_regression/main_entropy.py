import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
from regression import points_partition
from optimal_partition import fuzzy_optimal_partition
from correspondence_matrix import calc_correspondence_matrix, calc_reduced_correspondence_matrix
from fuzzy_min_entropy import fuzzy_min_entropy, fuzzy_min_entropy_t_crisp
from fuzzy_multivariate_regression import compute_u, fuzzy_plot_points_partition, fuzzy_plot_points_partition_coloured
from distrubution import calc_distribution


def read_data(file_name, rows, cols):
    # чтение входных данных
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=";")
        next(reader, None)  # пропустить заголовки
        data_str = [row for row in reader]

    # преобразование данных в числовой формат
    data_size = len(data_str)  # число точек
    assert(data_size == rows)
    data = [[] for _ in range(data_size)]
    for i in range(data_size):
        data[i] = list(map(float, data_str[i]))
        assert(len(data[i]) == cols)

    return np.array(data)


t_num = 5  # число моментов времени
compet_num = 38  # число компетенций
kpi_num = 4  # число ключевых показателей
burnout_num = 3  # число показателей выгорания
data_size = 219

x_ranges_num = 5  # число диапазонов взвешенной суммы

compet = [None for t in range(t_num)]
compet_files = ["compet_t" + str(i) + ".csv" for i in range(t_num)]
for t in range(t_num):
    compet[t] = read_data(compet_files[t], data_size, compet_num)

burnout = [None for t in range(t_num)]
burnout_files = ["burnout_t" + str(i) + ".csv" for i in range(t_num)]
for t in range(t_num):
    burnout[t] = read_data(burnout_files[t], data_size, burnout_num)

kpi = [None for t in range(t_num)]
kpi_files = ["kpi_t" + str(i) + ".csv" for i in range(t_num)]
for t in range(t_num):
    kpi[t] = read_data(kpi_files[t], data_size, kpi_num)


KPI_ind = 0  # интересующий показатель KPI
# берем для каждого сотрудника его компетенции в последний момент времени
data_x = np.array(compet[t_num - 1])
# берем для каждого сотрудника среднее KPI за все моменты времени
data_y = np.array([np.mean([kpi[t][i][KPI_ind] for t in range(t_num)]) for i in range(data_size)])
# вычисляем средний показатель выгорания, также усредненный по времени
#data_z_mean = np.array([np.mean([(burnout[t][i][0] + burnout[t][i][1] + burnout[t][i][2]) / 3\
#                            for t in range(t_num)]) for i in range(data_size)])
data_z = np.array([[np.mean([burnout[t][i][j] for t in range(t_num)]) for j in range(burnout_num)] for i in range(data_size)])


reg = LinearRegression().fit(data_x, data_y)
w0 = reg.coef_
lin_reg_R2 = r2_score(data_y, reg.predict(data_x))
print("LinR2 = ", lin_reg_R2)
print(w0)
# Вычисляем ошибку модели
rmse = math.sqrt(np.mean((data_y - reg.predict(data_x)) ** 2))
print("Lin_RMSE =", rmse)
# находим разбиение оси взвешенных сумм на интервалы
integral_x = np.dot(data_x, np.transpose(w0))  # интегральный показатель компетентности
t0 = points_partition(integral_x, data_y, x_ranges_num)

#J, mat = calc_correspondence_matrix(integral_x, data_y, t0)
J, mat = calc_reduced_correspondence_matrix(integral_x, data_y, data_z, t0)
np.set_printoptions(precision=5, suppress=True)
print(mat)
print("J =", J)

"""
iter_num = 200
lam = 0.01
t = fuzzy_optimal_partition(integral_x, data_y, x_ranges_num, t0, iter_num, lam)
J, mat = calc_reduced_correspondence_matrix(integral_x, data_y, data_z, t)
print(mat)
print("J =", J)
"""

# 5 классов
w0 = [ 2.42568, -0.01622, -1.55724, 0.61469,  0.49749,  2.28665,  0.77113,  0.93025,
  0.29054,  2.80141,  0.46901,  1.49558,  0.28327,  1.05572, -0.09754,  0.1322,
  0.67519,  0.50871,  2.68386,  4.0917,   1.75266,  1.87489, -1.68867,  1.48413,
  1.60866,  0.14488,  2.42032, -1.27706, 12.00537,  4.66233,  0.00421,  0.86921,
 -3.50815, -4.93438,  2.92068, -1.73377, -0.29395,  2.7637 ]
t0 = [53.84339218405032, 60.33215681343996, 74.15304632000989, 91.09094471075109]

w, t, c = fuzzy_min_entropy(data_x, data_y, data_z, x_ranges_num, 0, w0, t0)
#w, t, c = fuzzy_min_entropy(data_x, data_y, data_z, x_ranges_num, 20, w0, t0)
#w, t = fuzzy_min_entropy_t_crisp(data_x, data_y, data_z, x_ranges_num, w0)
integral_x = np.dot(data_x, np.transpose(w))  # интегральный показатель компетентности
J, mat = calc_reduced_correspondence_matrix(integral_x, data_y, data_z, t, verbose=True)
print(mat)
print("J =", J)
print("w =", list(w))
print("t =", list(t))
print("c =", c)
uu = compute_u(t, integral_x)
#fuzzy_plot_points_partition(integral_x, data_y, t, uu)
fuzzy_plot_points_partition_coloured(integral_x, data_y, t, uu, np.mean(data_z, axis=1))

# находим условные вероятности, что точка имеет определенный KPI, при условии
#   отнесения к определенной категории компетентности
# Заметим, что вероятность отнесения конкретной точки
#   к определенному значению KPI можно будет вычислить
#   по формуле полной вероятности, используя меры принадлежности
#   к категориям компетентности
d = calc_distribution(integral_x, data_y, data_z, t,
                      np.min(data_y), np.max(data_y), 100)
print(d)
