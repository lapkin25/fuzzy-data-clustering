import csv
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
from regression import *
from fuzzy_multivariate_regression import *
from optimal_partition import fuzzy_optimal_partition
from pwc_regression import multivariate_pwc_regression
from fuzzy_pwc_regression import fuzzy_multivariate_pwc_regression

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

x_ranges_num = 6  # число диапазонов взвешенной суммы
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

# определяет сумму квадратов отклонений для кусочно-постоянной регрессии
#   y на x, где x = w1*x1 + ... + wp*xp
def g(w):
    x = np.array([np.dot(data_x[i, :], w) for i in range(data_size)])
    y = data_y
    iter_num = 70
    lam = 0.03
    t0 = points_partition(x, y, x_ranges_num)

    #print(t0)
#       t = fuzzy_points_partition(x, y, x_ranges_num)
#       u = compute_u(t, x)
#    c = compute_c(u, y)
#    _, J, R2 = fuzzy_partition_summary(x, y, u)
#    print("J0 =", J)
#    partition = points_partition_given_avg(x, y, x_ranges_num, c)
#    u = compute_u(partition, x)
#    fuzzy_plot_points_partition(x, y, t, u)
#       _, J, R2 = fuzzy_partition_summary(x, y, u)
#       print("Оптимизация встроенным методом:  J =", J, " R2 =", R2)
    t = fuzzy_optimal_partition(x, y, x_ranges_num, t0, iter_num, lam)
    u = compute_u(t, x)
    _, J, R2 = fuzzy_partition_summary(x, y, u)
    print("Оптимизация градиентным методом:  J =", J, " R2 =", R2)

    #print(t)
    #fuzzy_plot_points_partition(x, y, t, u)
    return J

data_x = np.array(compet[0])
#x_transform = lambda x: math.log(x + 1)
#№v_x_transform = np.vectorize(x_transform)
#data_x = v_x_transform(data_x)
#print("data_x = ", data_x)
data_y = np.array([kpi[0][i][0] for i in range(data_size)])
#y_transform = lambda y: math.log(y)
#v_y_transform = np.vectorize(y_transform)
#data_y = v_y_transform(data_y)
#print("data_y = ", data_y)

reg = LinearRegression().fit(data_x, data_y)
w0 = reg.coef_ #np.array([0] * compet_num)
lin_reg_R2 = r2_score(data_y, reg.predict(data_x))
print("LinR2 = ", lin_reg_R2)

t0 =  [44.80370994383878, 66.54591318583549, 67.75642233952557, 82.8914899975313, 97.37910423875782]
w0 = [0.957013397157085, 1.4530371371926596, -1.9430914937781096, 1.8625149593641188, -1.8678561516532322, 1.3016197473626667, 0.8173777586177537, -1.3507362248026025, 0.8469441955659865, 3.9641431284661, 3.3231763521483213, -0.5925889684422266, 0.04780123134677039, 1.6093011981218293, -1.5754040423916504, 1.7539252135389027, 2.2207736436106384, 3.9145364487080765, 3.300458298829502, 1.1073750344183648, 2.763905833494696, -0.7462810301969082, 2.0153575473084113, -0.9248260311110151, 1.2699145395967022, 0.6926809695046481, 1.7143917864156577, -2.0667323345495476, 1.8742666398402588, 2.885207128385393, 0.22890997162607707, 2.3510663917103076, -2.3753208350494286, -5.159981827060327, 3.839323856254133, 0.24535954044919064, 2.076059960197496, 2.9980501925399894]
w, t = fuzzy_multivariate_pwc_regression(data_x, data_y, x_ranges_num, 5, w0, t0)
print(list(w))
print(list(t))

"""
#w, t, J = multivariate_pwc_regression(data_x, data_y, x_ranges_num)
w = multivariate_pwc_regression(data_x, data_y, x_ranges_num)
#res = minimize(f, w0, method='Nelder-Mead')
#res = minimize(g, w0, method='Powell') #tol=0.1, options={'disp': True, 'maxiter': 10})
#w = res.x
xx = np.array([np.dot(data_x[i, :], w) for i in range(data_size)])
t0 = points_partition(xx, data_y, x_ranges_num)
tt = fuzzy_optimal_partition(xx, data_y, x_ranges_num, t0, 100, 0.03)
uu = compute_u(tt, xx)
fuzzy_plot_points_partition(xx, data_y, tt, uu)
_, J, R2 = fuzzy_partition_summary(xx, data_y, uu)
print("Значение нечеткого функционала: ", J, " R2 =", R2)
"""

"""
data_x = np.array([1, -1, 0, -2, 3, 5])
data_y = np.array([1, 3, 1, -1, 0, -2])
t = points_partition(data_x, data_y, 3)
plot_points_partition(data_x, data_y, t)
_, _, R2 = partition_summary(data_x, data_y, t)
print("R2 =", R2)
"""

