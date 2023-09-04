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
from weighted_regression import calc_weighted_regression


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
burnout_num = 3  # число показателей выгорания
data_size = 219

x_ranges_num = 5  # число диапазонов взвешенной суммы
alpha = 1  # степенной параметр при компетенциях
beta = 1  # степенной параметр при взвешенной сумме

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

data_x = np.array(compet[4])
#x_transform = lambda x: math.log(x + 1)
#№v_x_transform = np.vectorize(x_transform)
#data_x = v_x_transform(data_x)
#print("data_x = ", data_x)
data_y = np.array([kpi[4][i][0] for i in range(data_size)])
data_z = np.array([(burnout[4][i][0] + burnout[4][i][1] + burnout[4][i][2]) / 3 for i in range(data_size)])
data_burnout = np.array(burnout[4])
#y_transform = lambda y: math.log(y)
#v_y_transform = np.vectorize(y_transform)
#data_y = v_y_transform(data_y)
#print("data_y = ", data_y)

reg = LinearRegression().fit(data_x, data_y)
w0 = reg.coef_ #np.array([0] * compet_num)
lin_reg_R2 = r2_score(data_y, reg.predict(data_x))
print("LinR2 = ", lin_reg_R2)
print(w0)

#t0 = [42.02647369826323, 67.30204570833794, 67.30204570833794, 81.88765091530608, 98.39180259081934]
#w0 = [1.0668125285830576, 1.4223549588361353, -2.213422029742069, 1.6940743180553293, -1.5710809480910235, 1.27368896851416, 0.8072842279499194, -1.288554413316511, 0.744753381233767, 3.9266537391628935, 3.407727366074928, -0.5131851231698, 0.15143493145343292, 1.5472700616213129, -1.569737753880028, 1.9238894090762408, 2.1853715770155144, 3.7370668031679464, 3.3571034971736777, 1.0807770747162386, 2.778926054531606, -0.7827907856707941, 2.029568278716351, -0.9185416050204861, 1.2583903589756336, 0.5871690123123022, 1.707156858493731, -1.9521261240721206, 1.5603569259508698, 2.976530344806123, 0.6978625270364278, 2.144396827461618, -1.9367619243081855, -5.603657524200582, 4.161417641236367, -0.02997383716000733, 2.46213814044579, 2.699737314424496]

t0 = [55.30664095988004, 55.30664095988004, 83.12673775483862, 97.20512631222665]
w0 = [2.5702968508591724, 1.3011614299853123, -3.2864385729919046, 1.7424365572379874, -2.7585981369237422, 2.0098608175378163, 1.2100757792561156, 0.05718422562419469, 0.0596658670958462, 3.5805007756185763, 2.3074061505350647, 0.7091847063028334, 1.7384970958506114, 0.21376960384420174, -2.637313020917647, 1.6667443330062859, 1.9661783044089285, 3.8361822825150442, 2.5858081331702727, 2.5811354402871407, 2.5565832212381303, -0.5002250646455696, 1.1253851197171558, -0.8235397553767552, 1.4333472841418844, 0.42018398083114566, 0.9215882978024285, -0.26990825606495666, 4.204417466713239, 3.6404496995730806, -0.35029394831264243, 2.256080180952055, -0.9523585419112787, -5.2604934961864735, 2.741696336494361, -1.780444130415401, 0.9768545770140458, 3.7948862941781796]

w, t = fuzzy_multivariate_pwc_regression(data_x, data_y, x_ranges_num, 0, w0, t0)
print(list(w))
print(list(t))


#w, t, J = multivariate_pwc_regression(data_x, data_y, x_ranges_num)
#w = multivariate_pwc_regression(data_x, data_y, x_ranges_num)
#res = minimize(f, w0, method='Nelder-Mead')
#res = minimize(g, w0, method='Powell') #tol=0.1, options={'disp': True, 'maxiter': 10})
#w = res.x
xx = np.array([np.dot(data_x[i, :], w) for i in range(data_size)])
#t0 = points_partition(xx, data_y, x_ranges_num)
#tt = fuzzy_optimal_partition(xx, data_y, x_ranges_num, t0, 100, 0.03)
uu = compute_u(t, xx)
#fuzzy_plot_points_partition(xx, data_y, t, uu)
fuzzy_plot_points_partition_coloured(xx, data_y, t, uu, data_z)
_, J, R2 = fuzzy_partition_summary(xx, data_y, uu)
print("Значение нечеткого функционала: ", J, " R2 =", R2)

calc_weighted_regression(data_burnout, data_y, uu[:, 0])

"""
data_x = np.array([1, -1, 0, -2, 3, 5])
data_y = np.array([1, 3, 1, -1, 0, -2])
t = points_partition(data_x, data_y, 3)
plot_points_partition(data_x, data_y, t)
_, _, R2 = partition_summary(data_x, data_y, t)
print("R2 =", R2)
"""

