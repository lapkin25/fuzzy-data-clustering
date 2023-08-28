# Поиск функциональной связи между двумя одномерными рядами данных

import matplotlib.pyplot as plt
import numpy as np

# data_x - одномерный массив, data_y - одномерный массив
# y_min, y_max - диапазон возможных значений y
# y_classes_num - количество классов
def process1d (data_x, data_y, y_min, y_max, y_classes_num, t, intercept = 0, C = 1):
    plt.plot(data_x, data_y, 'ro')
    min_x = np.min(data_x)
    max_x = np.max(data_x)
    plt.plot([min_x, max_x], [C * min_x + intercept, C * max_x + intercept])
    for i in range(len(t)):
        plt.plot([t[i], t[i]], [0, 100], 'g')
    for i in range(y_classes_num):
        plt.plot([min_x, max_x], [i * (y_max - y_min) / y_classes_num, i * (y_max - y_min) / y_classes_num], 'b')
    plt.xlabel("Взвешенная сумма ожиданий")
    plt.ylabel("Показатель выгорания")
    plt.show()
