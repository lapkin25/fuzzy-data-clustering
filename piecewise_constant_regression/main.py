import csv
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from regression import *

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

    return data


t_num = 5  # число моментов времени
compet_num = 38  # число компетенций
kpi_num = 4  # число ключевых показателей
data_size = 219

x_ranges_num = 7  # число диапазонов взвешенной суммы
alpha = 1  # степенной параметр при компетенциях
beta = 1  # степенной параметр при взвешенной сумме

compet = [None for t in range(t_num)]
compet_files = ["compet_t" + str(i) + ".csv" for i in range(t_num)]
for t in range(t_num):
    compet[t] = read_data(compet_files[t], data_size, compet_num)

kpi = [None for t in range(t_num)]
kpi_files = ["kpi_t" + str(i) + ".csv" for i in range(t_num)]
for t in range(t_num):
    kpi[t] = read_data(kpi_files[t], data_size, kpi_num)


# определяет сумму квадратов отклонений для кусочно-постоянной регрессии
#   y на x, где x = w1*x1 + ... + wp*xp
def f(w):
    x = np.array([np.dot(data_x[i, :] ** alpha, w) ** beta for i in range(data_size)])
    y = data_y
    partition = points_partition(x, y, x_ranges_num)
    plot_points_partition(x, y, partition)
    _, J, R2 = partition_summary(x, y, partition)
    #print("w =", w)
    print("J =", J, " R2 =", R2)
    return J


data_x = np.array(compet[0])
#print("data_x = ", data_x)
data_y = np.array([kpi[0][i][0] for i in range(data_size)])
#print("data_y = ", data_y)

reg = LinearRegression().fit(data_x, data_y)
w0 = reg.coef_ #np.array([0] * compet_num)
lin_reg_R2 = r2_score(data_y, reg.predict(data_x))
print("LinR2 = ", lin_reg_R2)
res = minimize(f, w0, method='Nelder-Mead')



"""
data_x = np.array([1, -1, 0, -2, 3, 5])
data_y = np.array([1, 3, 1, -1, 0, -2])
t = points_partition(data_x, data_y, 3)
plot_points_partition(data_x, data_y, t)
_, _, R2 = partition_summary(data_x, data_y, t)
print("R2 =", R2)
"""

