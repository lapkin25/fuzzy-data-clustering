import csv
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from process1d import process1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def read_data(file_name, x_num, y_num):
    # чтение входных данных
    with open(file_name) as fp:
        reader = csv.reader(fp, delimiter=";")
        next(reader, None)  # пропустить заголовки
        data_str = [row for row in reader]

    # преобразование данных в числовой формат
    data_size = len(data_str)  # число точек
    data = [[] for _ in range(data_size)]
    for i in range(data_size):
        data[i] = list(map(float, data_str[i]))
        assert(len(data[i]) == x_num + y_num)

    data_x = np.array([[data[i][j] for j in range(x_num)] for i in range(data_size)])
    data_y = np.array([[data[i][x_num + j] for j in range(y_num)] for i in range(data_size)])

    return data_x, data_y


def predict_binary(x_data, y_data):
    try:
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(x_data, y_data)
        y_predicted = reg.predict(x_data)
    except ValueError:
        y_predicted = y_data   
#    print(y_predicted)
    return y_predicted


# y_min, y_max - диапазон возможных значений y
y_min = 0
y_max = 100
y_classes_num = 5  # число диапазонов y

# возвращает номер класса от 0 до 9
def y_class(y):
    return int(y / 10) if y < 100 else 9

y_class_v = np.vectorize(y_class)


def predict_multinomial(x_data, y_data):
    y_data_bin = []
    for i in range(1, y_classes_num):
        f_i = np.vectorize(lambda n: int(n >= i))
        y_data_bin.append(f_i(y_data))
    #print(y_data_bin)
    y_predicted = []
    for i in range(y_classes_num - 1):
        y_predicted.append(predict_binary(x_data, y_data_bin[i]))
    # вычисляем сумму регрессий с номерами от 0 до y_classes_num - 1
    y = y_predicted[0].copy()
    for i in range(1, y_classes_num - 1):
        y += y_predicted[i]
    print("Исходные значения:")
    print(y_data)
    print("Предсказанные значения:")
    print(y)
    print("Средняя разница:", np.mean(abs(y - y_data)))


x_num = 58 #29
y_num = 3

def cost_functional (data_x, data_y):
    p = data_x.argsort()
    data_x_sorted = data_x[p]
    data_y_sorted = data_y[p]

    data_size = data_x_sorted.shape[0]
    u = [[0 for k in range(y_classes_num)] for i in range(data_size)]
    # формируем матрицу u[i][k] - принадлежность i-й точки к k-му классу
    for i in np.nditer(np.arange(data_size)):
        #print(data_x_sorted[i], data_y_sorted[i])
        for k in range(y_classes_num):
            # (x1, y1), (x2, y2), (x3, y3) - координаты вершин k-го треугольника    
            x2 = (k + 0.5) * (y_max - y_min) / y_classes_num
            y2 = 1
            x1 = x2 - (y_max - y_min) / y_classes_num
            y1 = 1 if k == 0 else 0
            x3 = x2 + (y_max - y_min) / y_classes_num
            y3 = 1 if k == y_classes_num - 1 else 0
            x = data_y_sorted[i]
            if x <= x1 or x >= x3:
                mu = 0
            elif x >= x1 and x < x2:
                mu = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            elif x >= x2 and x < x3:
                mu = y2 + (y3 - y2) * (x - x2) / (x3 - x2)
            u[i][k] = mu
#    print("u = ", u)
#    print("y_sorted = ", data_y_sorted)


    # вместо нечетких чисел используем четкое отнесение точек к диапазонам по y
 #   u = [[0 for k in range(y_classes_num)] for i in range(data_size)]
 #   for i in range(data_size):
 #       for k in range(y_classes_num):
 #           if data_y_sorted[i] > (y_max - y_min) / y_classes_num * k and data_y_sorted[i] <= (y_max - y_min) / y_classes_num * (k + 1):
 #               u[i][k] = 1


    # вычисляем t[k], k = 0..y_classes_num-2 - границы между k-м и (k+1)-м диапазонами по x
    t = [0 for k in range(y_classes_num - 1)]
    for k in range(y_classes_num - 1):
        # слева и справа от крайних точек излома кусочно-линейной функции корня нет
        for j in range(data_size - 1):
            # ищем корень на промежутке [x[j], x[j+1]]
            # обращаемся к коэффициентам u[i][k], u[i][k+1] на промежутке [x[j], x[j+1]]
#            x1 = x[j]
            # линейное уравнение a * t + b = 0
            coeff_a = 0
            coeff_b = 0
            for i in range(data_size):
                if i <= j:
                    # когда t принадлежит промежутку [x[j], x[j+1]], имеем t - x[i] >= 0
                    coeff_a += u[i][k + 1]
                    coeff_b -= u[i][k + 1] * data_x_sorted[i]
                else:
                    # когда t принадлежит промежутку [x[j], x[j+1]], имеем t - x[i] <= 0
                    coeff_a += u[i][k]
                    coeff_b -= u[i][k] * data_x_sorted[i]
 #           x2 = x[j + 1]
            if coeff_a == 0:
                continue
            t_sol = - coeff_b / coeff_a
            if t_sol >= data_x_sorted[j] and t_sol <= data_x_sorted[j + 1]:
                t[k] = t_sol
 
    # вычисляем функционал стоимости
    J = 0
    for i in range(data_size):
        for k in range(y_classes_num):
            term = 0  # квадрат расстояния от точки x[i] до k-го диапазона [t[k-1], t[k]]
            if k > 0 and data_x_sorted[i] < t[k - 1]:
                term = (data_x_sorted[i] - t[k - 1]) ** 2
            elif k < y_classes_num - 1 and data_x_sorted[i] > t[k]:
                term = (data_x_sorted[i] - t[k]) ** 2
            J += term * u[i][k]
    J /= data_size

    # вычисляем матрицу соответствий
    s = [[0 for j in range(y_classes_num)] for i in range(y_classes_num)]
    for i in range(data_size):
        # находим x_class как наибольшее j, для которого x > t[j-1]
        x_class = 0
        for j in range(1, y_classes_num):
            if data_x_sorted[i] > t[j - 1]:
                x_class = j
        for k in range(y_classes_num):
            s[x_class][k] += u[i][k]
    for j in range(y_classes_num):
        sum_s = 0
        for k in range(y_classes_num):
            sum_s += s[j][k]
        if sum_s > 0:
            for k in range(y_classes_num):
                s[j][k] /= sum_s

    return J, t, s


def f(w):
    x = np.dot(data_x, w)
    J, _, _ = cost_functional(x, data_y[:, y_ind])
    sq_norm_w = np.linalg.norm(w) ** 2 
    J_tilde = J / sq_norm_w
    print(J_tilde)
    return J_tilde


data_x, data_y = read_data("data.csv", x_num, y_num)

# статистика x_k
data_size = data_x.shape[0]
print(data_size)
for k in range(x_num):
    cnt_pos = 0
    cnt_neg = 0
    sum_pos = 0
    sum_neg = 0
    sum_sqr_pos = 0
    sum_sqr_neg = 0
    for i in range(data_size):
        if data_x[i][k] > 0:
            cnt_pos += 1
            sum_pos += data_x[i][k]
            sum_sqr_pos += data_x[i][k] ** 2
        elif data_x[i][k] < 0:
            cnt_neg += 1
            sum_neg += data_x[i][k]
            sum_sqr_neg += data_x[i][k] ** 2
    mean_pos = sum_pos / cnt_pos if cnt_pos > 0 else 0
    rms_pos = math.sqrt(sum_sqr_pos / cnt_pos - mean_pos ** 2) if cnt_pos > 0 else 0
    mean_neg = sum_neg / cnt_neg if cnt_neg > 0 else 0
    rms_neg = math.sqrt(sum_sqr_neg / cnt_neg - mean_neg ** 2) if cnt_neg > 0 else 0
    print("k =", k, "mean_pos =", mean_pos, "rms_pos = ", rms_pos, "mean_neg = ", mean_neg, "rms_neg = ", rms_neg)        
        

for y_ind in range(y_num):
    reg = LinearRegression().fit(data_x, data_y[:, y_ind])
    predicted_y = reg.predict(data_x)
    print("=============")
    print("Y", y_ind, sep='')
    print("=============")
    print("R^2 = ", reg.score(data_x, data_y[:, y_ind]))
    #print(np.column_stack((data_y[:, y_ind], predicted_y)))
    print('root_mean_squared_error : ', math.sqrt(mean_squared_error(data_y[:, y_ind], predicted_y)))
    print('mean_absolute_error : ', mean_absolute_error(data_y[:, y_ind], predicted_y))
    print('w =', reg.coef_ / np.linalg.norm(reg.coef_))

    J, t, confusion_matrix = cost_functional(predicted_y, data_y[:, y_ind])
    print("J = ", J)
    print("t = ", (t - reg.intercept_) / np.linalg.norm(reg.coef_))
    print("Матрица соответствий: ", confusion_matrix)
#    process1d(predicted_y, data_y[:, y_ind], 0, 100, y_classes_num, t)
    process1d(np.dot(data_x, reg.coef_) / np.linalg.norm(reg.coef_), data_y[:, y_ind], 0, 100, y_classes_num, (t - reg.intercept_) / np.linalg.norm(reg.coef_), reg.intercept_, np.linalg.norm(reg.coef_))

    w0 = reg.coef_
    print("J~ = ", f(w0))
#    res = minimize(f, w0, method='Nelder-Mead')
    w_opt = res.x
    w_opt /= np.linalg.norm(w_opt)
    print("J~_opt = ", f(w_opt))
    print("w_opt = ", w_opt)
    predicted_y_opt = np.dot(data_x, w_opt)   
    _, t, confusion_matrix = cost_functional(predicted_y_opt, data_y[:, y_ind])
    print("Матрица соответствий:", confusion_matrix)
    print("t = ", t)

    plt.plot(predicted_y_opt, data_y[:, y_ind], 'ro')
#    plt.plot([0, 100], [0, 100])
    for i in range(len(t)):
        plt.plot([t[i], t[i]], [0, 100], 'g')
    for i in range(y_classes_num):
        plt.plot([min(predicted_y_opt), max(predicted_y_opt)], [i * (y_max - y_min) / y_classes_num, i * (y_max - y_min) / y_classes_num], 'b')
    plt.show()
