import csv
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
from regression import *
from fuzzy_multivariate_regression import *
from optimal_partition import fuzzy_optimal_partition  #, calc_a
from pwc_regression import multivariate_pwc_regression
from fuzzy_pwc_regression import fuzzy_multivariate_pwc_regression
from weighted_regression import calc_weighted_regression,\
    plot_weighted_regression


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

x_ranges_num = 6  # число диапазонов взвешенной суммы
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

"""
data_x = np.array(compet[4])
#x_transform = lambda x: math.log(x + 1)
#v_x_transform = np.vectorize(x_transform)
#data_x = v_x_transform(data_x)
#print("data_x = ", data_x)
data_y = np.array([kpi[4][i][0] for i in range(data_size)])
data_z = np.array([(burnout[4][i][0] + burnout[4][i][1] + burnout[4][i][2]) / 3 for i in range(data_size)])
data_burnout = np.array(burnout[4])
#y_transform = lambda y: math.log(y)
#v_y_transform = np.vectorize(y_transform)
#data_y = v_y_transform(data_y)
#print("data_y = ", data_y)
"""

KPI_ind = 0  # интересующий показатель KPI
# берем для каждого сотрудника его компетенции в последний момент времени
#data_x = np.array(compet[t_num - 1])

# берем компетенции в начальный момент времени
data_x = np.array(compet[0])

# берем для каждого сотрудника среднее KPI за все моменты времени
#data_y = np.array([np.mean([kpi[t][i][KPI_ind] for t in range(t_num)]) for i in range(data_size)])

# берем KPI в начальный момент времени
data_y = np.array([kpi[0][i][KPI_ind] for i in range(data_size)])

# вычисляем средний показатель выгорания, также усредненный по времени
data_z = np.array([np.mean([(burnout[t][i][0] + burnout[t][i][1] + burnout[t][i][2]) / 3\
                            for t in range(t_num)]) for i in range(data_size)])


reg = LinearRegression().fit(data_x, data_y)
w0 = reg.coef_ #np.array([0] * compet_num)
lin_reg_R2 = r2_score(data_y, reg.predict(data_x))
print("LinR2 = ", lin_reg_R2)
print(w0)
t0 = points_partition(np.dot(data_x, np.transpose(w0)), data_y, x_ranges_num)
# Вычисляем ошибку модели
rmse = math.sqrt(np.mean((data_y - reg.predict(data_x)) ** 2))
print("Lin_RMSE =", rmse)


# Строим линейную регрессию с учетом выгорания
#print(data_x)
#print(data_z)
data_x_ext = np.c_[ data_x,\
    np.array([np.mean([burnout[t][i][0] for t in range(t_num)]) for i in range(data_size)]),\
    np.array([np.mean([burnout[t][i][1] for t in range(t_num)]) for i in range(data_size)]),\
    np.array([np.mean([burnout[t][i][2] for t in range(t_num)]) for i in range(data_size)]) ]
#print(data_x_ext)

reg = LinearRegression().fit(data_x_ext, data_y)
rmse = math.sqrt(np.mean((data_y - reg.predict(data_x_ext)) ** 2))
print("Lin_Extended_RMSE =", rmse)


# для среднего KPI1, m = 7 => J = 18053
#w0 = [0.7315991853297311, 1.606088425459402, -1.5007441840362965, 1.0823759889172127, -2.5536752008551917, 1.72524282633304, 1.15987597115423, -0.47578070966549063, 0.1511344979328968, 3.904647071382179, 1.2857101750027393, -0.19811808194350997, 0.7915742263408108, 1.5733021257159985, -2.0711862887888834, 1.7869647132553084, 0.9825623476299983, 3.8402252890103967, 1.6522344659870631, 2.727450908892836, 1.0030336313951382, 0.41007524107323073, 0.009780771758801527, -0.21008321018582288, 0.8340017386160649, 1.0047322032336221, -0.6175874973810114, 0.35496968251942546, 1.1372800240105978, 4.056994033839959, 0.68952028570449, 1.9392672369844222, -1.3290252597146934, -4.453504007265609, 2.1983647518309373, 1.3953636393206463, 0.7481236333763723, 0.751428722906295]
#t0 = [48.92530092318541, 48.92530092318542, 59.66120080564403, 67.33216124660905, 77.40287566361853, 87.91493692741578]

# для среднего KPI1, m = 6 => J = 18863
#w0 = [0.35102202469718624, 1.3267686935112049, -1.602607777442623, 1.5876620154082224, -1.9012313397234346, 0.9306288551834481, 1.3753441958088466, -1.0204356349794816, 0.003574038804005934, 3.0643038166672367, 2.8212610396857256, -0.3594912861259836, 0.7708979641822872, 1.3792524319000974, -1.1406357528147477, 1.119458320775773, 1.630513157380746, 3.809910192580168, 2.6716640397737086, 0.9764811499861118, 2.139506575456032, -1.293072953270076, 1.2764474850643635, -0.5696340025434599, 1.0727221652971284, 0.41911583354388643, 0.7026103150866185, -0.3742022442618117, 4.991442232950911, 0.7501645189823558, -0.687735641752203, 2.186974086603609, -1.4541587738830006, -2.0895280992175955, 1.7689784432099414, -0.9829001006719671, 0.3681240105328023, 1.4900036119461502]
#t0 = [41.11829475664042, 62.362611268486134, 62.362611268486134, 77.33570967040534, 87.64618257284772]

# для среднего KPI2, m = 6 => J = 22065
#w0 = [0.22105487608770236, 1.368846784861191, -1.8472156503183272, 0.7238453050164428, -2.1392445078205573, 2.1711779121053696, 2.2443881908494445, 1.0671028699856653, 0.28729044607059673, 1.8888417183505217, 1.8597511462412462, -1.2483689619648315, 1.4822574589803679, 0.12535497594753658, -0.056537396160046584, 2.0554466447642095, 0.4589784444985735, 3.6796699542666165, 0.39992163583151885, 1.4167809261731563, 2.990817636463288, 0.4945585924390337, 1.043903747021469, -0.8971661764853013, 0.9670803039972226, 0.2237501055072635, 0.47110306613180764, -0.055888673093403334, 3.370982656896399, 3.34841784283777, -1.3271482645102084, 0.9816695004502972, -0.8335962719820906, -0.945856078376565, 0.16819896533423595, -0.864972958255166, 1.5977968580245163, 1.6279219702471508]
#t0 = [49.58135362841478, 57.42075009296823, 64.21518129023433, 79.28578436181043, 90.1889901202023]

# для среднего KPI3, m = 6 => J = 23009
#w0 = [-1.8811826288242124, 2.960755230218461, -3.2116112913471047, 2.912443407962199, -2.561954811011203, 4.657604495802157, -0.35669275983237064, -1.3695184229380952, 2.459719226497746, 3.039875762565631, 1.9936197813810945, -1.896458910004704, 2.567042601768098, -0.2441185427921764, -0.8230605969606137, 2.4951995808647127, 1.434954344056368, 3.685889966864712, 1.86853586103755, 1.6692617557239935, 2.7793371155748647, -1.5232016301621825, 0.2761971401470617, -1.5536631092370092, 2.0834728315096975, 0.5924449284188571, 0.8027132988156208, -0.009118319796385228, 1.578533916327386, 6.470428976214293, -3.507207893527386, 1.0301289445889934, 3.0272679980821633, -0.8529749192458799, 0.20036266660978047, -1.952443874579604, -1.603668506722932, 4.7200038079548365]
#t0 = [62.67939729944656, 62.80031569755209, 71.27809519367747, 87.67404466429113, 103.55807603839686]

# для среднего KPI4, m = 6 => J = 19183
# w0 = [0.36039390080229056, 1.5516372277068606, -2.811332459853894, 1.4217553248231614, -2.2026359251023364, 1.5415808870543566, 0.5515312076322817, 0.23350997519858352, 0.3788593160453449, 2.8106210191765615, 1.1200349604105446, -1.8520825444461786, 1.679809268873391, 1.3838980376363468, -1.224144833365135, 2.305175995672064, 2.878225308743788, 3.9319971105947116, 0.7531193112452502, 0.7588148817447969, 1.0497888398759165, -1.6907454061411349, 1.047337940998625, -1.0470021249662531, 2.088667836504559, 1.8546069683472777, 1.0576272696182596, 0.44718508685061337, 3.7256295917969813, 2.035098132131174, 0.37261182828508765, 1.8549527138721729, -0.5670285185281299, -1.8316011505066212, 0.34509253288786357, -1.4230998946233533, -1.0878582654352933, 0.955422659247028]
# t0 = [52.18789933095373, 52.18789933095373, 63.42661492243427, 77.16173826129062, 80.5110737076746]


#t0 = [42.02647369826323, 67.30204570833794, 67.30204570833794, 81.88765091530608, 98.39180259081934]
#w0 = [1.0668125285830576, 1.4223549588361353, -2.213422029742069, 1.6940743180553293, -1.5710809480910235, 1.27368896851416, 0.8072842279499194, -1.288554413316511, 0.744753381233767, 3.9266537391628935, 3.407727366074928, -0.5131851231698, 0.15143493145343292, 1.5472700616213129, -1.569737753880028, 1.9238894090762408, 2.1853715770155144, 3.7370668031679464, 3.3571034971736777, 1.0807770747162386, 2.778926054531606, -0.7827907856707941, 2.029568278716351, -0.9185416050204861, 1.2583903589756336, 0.5871690123123022, 1.707156858493731, -1.9521261240721206, 1.5603569259508698, 2.976530344806123, 0.6978625270364278, 2.144396827461618, -1.9367619243081855, -5.603657524200582, 4.161417641236367, -0.02997383716000733, 2.46213814044579, 2.699737314424496]

# для t = 0
#t0 = [55.30664095988004, 55.30664095988004, 83.12673775483862, 97.20512631222665]
#w0 = [2.5702968508591724, 1.3011614299853123, -3.2864385729919046, 1.7424365572379874, -2.7585981369237422, 2.0098608175378163, 1.2100757792561156, 0.05718422562419469, 0.0596658670958462, 3.5805007756185763, 2.3074061505350647, 0.7091847063028334, 1.7384970958506114, 0.21376960384420174, -2.637313020917647, 1.6667443330062859, 1.9661783044089285, 3.8361822825150442, 2.5858081331702727, 2.5811354402871407, 2.5565832212381303, -0.5002250646455696, 1.1253851197171558, -0.8235397553767552, 1.4333472841418844, 0.42018398083114566, 0.9215882978024285, -0.26990825606495666, 4.204417466713239, 3.6404496995730806, -0.35029394831264243, 2.256080180952055, -0.9523585419112787, -5.2604934961864735, 2.741696336494361, -1.780444130415401, 0.9768545770140458, 3.7948862941781796]

# для t = 4, m = 5
#t0 = [62.33182587947523, 62.33182587947524, 95.57739081570118, 105.80945551244034]
#w0 = [1.5210899931781685, 2.0423368903325705, -2.664880879801911, 2.3793738236629696, -1.5803277920941325, 2.2246046963090613, 0.6106369115664912, -1.041931764271053, 0.8740657857816169, 3.4824141430747413, 1.8007046307501038, 0.10797882048972736, 2.3310859409334497, 1.2016963148755817, -2.067241684028803, 0.1349586227463114, 1.1335996661773893, 3.5544660623110067, 4.786471146472953, 0.3041510796578768, 2.9180407820579792, -1.8055845311518792, 1.939022652024267, -0.6935598744426983, 1.4300512371098029, 1.6239075785720383, 0.6076969119343267, -0.8061803759105762, 5.010083515257503, 3.874997263525974, -0.625246749291251, 2.4011415015079933, -0.2729630623249757, -3.133276630917914, 2.1558748366669542, -0.3026252183615371, -1.2240691186159767, 1.8374834340340342]


w, t = fuzzy_multivariate_pwc_regression(data_x, data_y, x_ranges_num, 12, w0, t0)
print(list(w))
print(list(t))


"""
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
"""

x1 = np.array([np.dot(data_x[i, :], w) for i in range(data_size)])
#print(np.min(x1), np.max(x1))
u1 = compute_u(t, x1)

fuzzy_plot_points_partition_coloured(x1, data_y, t, u1, data_z)
avg_y, J, R2 = fuzzy_partition_summary(x1, data_y, u1)
print("Значение нечеткого функционала: ", J, " R2 =", R2)
print("avg_y =", avg_y)

fout = open("compet.txt", "w")
for i in np.arange(data_size):
    fout.write('{0:f}\n'.format(x1[i]))
fout.close()


#w, w0 = calc_weighted_regression(data_burnout, data_y, uu[:, 0])
#print("w =", w)
#print("w0 =", w0)


# вычисляем зависимость от выгорания (общую для всех интервалов компетентности)
#a = calc_a(t, np.min(x1), np.max(x1))
diff_y = np.zeros(data_size)
for i in range(data_size):
    diff_y[i] = data_y[i] - np.dot(u1[i, :], avg_y)

print("diff_y =", diff_y)

data_burnout = np.array([[burnout[0][i][m] for m in range(burnout_num)] for i in range(data_size)])
reg = LinearRegression().fit(data_burnout, diff_y)
w = reg.coef_
w0 = reg.intercept_
lin_reg_R2 = r2_score(diff_y, reg.predict(data_burnout))
print("LinR2 = ", lin_reg_R2)
print("coef =", w)
print("intercept =", w0)

wsum_x = reg.predict(data_burnout)
plt.scatter(wsum_x, diff_y)
plt.plot([np.min(wsum_x), np.max(wsum_x)], [np.min(wsum_x), np.max(wsum_x)])
plt.xlabel("Integral burnout indicator")
plt.ylabel("diff KPI")
plt.show()

"""
# выгорание по всем моментам времени
full_data_burnout = []
for tm in range(t_num):
    #print(burnout[tm])
    if tm == 0:
        full_data_burnout = burnout[0]
    else:
        full_data_burnout = np.concatenate((full_data_burnout, burnout[tm]), axis=0)
    #print(full_data_burnout)
# KPI по всем моментам времени
full_data_kpi = []
for tm in range(t_num):
    new_kpi = np.array([kpi[tm][i][KPI_ind] for i in range(data_size)])
    if tm == 0:
        full_data_kpi = new_kpi
    else:
        full_data_kpi = np.concatenate((full_data_kpi, new_kpi), axis=0)
# мера принадлежности k-му диапазону компетенций по всем моментам времени
full_u = []
for tm in range(t_num):
    if tm == 0:
        full_u = u1
    else:
        full_u = np.concatenate((full_u, u1), axis=0)

#print(full_data_burnout.shape)
"""


# Усредняем показатели по времени
# среднее выгорание по времени
for tm in range(t_num):
    if tm == 0:
        avg_burnout = burnout[0]
    else:
        avg_burnout += burnout[tm]
avg_burnout /= t_num
# средний KPI по времени
for tm in range(t_num):
    new_kpi = np.array([kpi[tm][i][KPI_ind] for i in range(data_size)])
    if tm == 0:
        avg_kpi = new_kpi
    else:
        avg_kpi += new_kpi
avg_kpi /= t_num


"""
# строит график для k-го диапазона компетенций
def plot_kpi_burnout_for_compet_range(k):
    w, w0 = calc_weighted_regression(data_burnout, data_y, uu[:, k])
    plot_weighted_regression(data_burnout[uu[:, k] != 0], data_y[uu[:, k] != 0], uu[uu[:, k] != 0, k], w, w0)
"""

"""
# строит график для k-го диапазона компетенций
def plot_kpi_burnout_for_compet_range(k):
    w, w0 = calc_weighted_regression(full_data_burnout, full_data_kpi, full_u[:, k])
    plot_weighted_regression(full_data_burnout[full_u[:, k] != 0],
                             full_data_kpi[full_u[:, k] != 0],
                             full_u[full_u[:, k] != 0, k], w, w0)
"""

# строит график для k-го диапазона компетенций
def plot_kpi_burnout_for_compet_range(k):
    w, w0 = calc_weighted_regression(avg_burnout, avg_kpi, u1[:, k])
    print("w =", w, "w0 =", w0)
    plot_weighted_regression(avg_burnout[u1[:, k] != 0],
                             avg_kpi[u1[:, k] != 0],
                             u1[u1[:, k] != 0, k], w, w0)


#plot_weighted_regression(data_burnout[uu[:, 0] != 0], data_y[uu[:, 0] != 0], uu[uu[:, 0] != 0, 0], w, w0)
plot_kpi_burnout_for_compet_range(1)


# TODO: найти R^2, то есть сумму квадратов отклонений модели от данных, деленную на дисперсию




"""
data_x = np.array([1, -1, 0, -2, 3, 5])
data_y = np.array([1, 3, 1, -1, 0, -2])
t = points_partition(data_x, data_y, 3)
plot_points_partition(data_x, data_y, t)
_, _, R2 = partition_summary(data_x, data_y, t)
print("R2 =", R2)
"""

