# Множественная кусочно-постоянная регрессия

from regression import *

# Находит оптимальный весовой множитель
# i-я точка в момент времени t имеет координату x_i(t) = z_i + v_i * t
# Функция возвращает t, при котором функционал кусочно-постоянной регрессии
#   для точек (x_i(t), y_i) при числе диапазонов m достигает минимума
def find_optimal_weight(v_unsorted, z_unsorted, y_unsorted, m):
    # сортируем точки по возрастанию v[i]
    ind_p = v_unsorted.argsort()  # находим перестановку индексов
    v = v_unsorted[ind_p]
    z = z_unsorted[ind_p]
    y = y_unsorted[ind_p]

    # начальный (при малых t) порядок точек: точки по убыванию v[i]
    # находим начальное оптимальное разбиение на диапазоны
    s = points_ordered_partition(y, m)

    # дальше все точки по очереди меняются местами с своим соседом справа
    # алгоритм мог бы основываться на допущении, что при перестановке
    #   точек внутри диапазона текущее разбиение на диапазоны
    #   останется оптимальным и для нового порядка точек
    #   Однако можно подобрать контрпример к этому утверждению.
    # Остается перебирать все точки пересечения прямых x_i(t)
    #   и для каждого момента времени находить оптимальное разбиение,
    #   что потребует O(n^4 * m) времени - слишком много
    pass


# Находит оптимальный весовой множитель
#   для заданного разбиения на диапазоны
def find_optimal_weight_given_partition(v, z, y, s):
    pass


data_v = np.array([1, 3, 1.5, 2, 1, 5])
data_s = np.array([1, 5, -2, 0, 1, -1])
data_y = np.array([1, 3, 1, -1, 0, -2])
w = find_optimal_weight(data_v, data_s, data_y, 3)
#t = points_partition(data_x, data_y, 3)
#plot_points_partition(data_x, data_y, t)
#_, _, R2 = partition_summary(data_x, data_y, t)
#print("R2 =", R2)
