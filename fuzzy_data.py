import csv
import random

class FuzzyNumber:
    """
    Трапециевидное нечеткое число

    Поля:
        c1 (float): левый центр
        c2 (float): правый центр
        c1 <= c2
        l (float): левый разброс
        r (float): правый разброс
    """

    def __init__(self, c1, c2, l, r):
        self.c1 = c1
        self.c2 = c2
        self.l = l
        self.r = r

    def __str__(self):
        return f"(c1 = {self.c1}, c2 = {self.c2}, l = {self.l}, r = {self.r})"

    def __repr__(self):
        return self.__str__()

def sqr_distance(a, b):
    """
    Вычисляет квадрат евклидова расстояния между векторами a и b
    """
    assert(len(a) == len(b))
    n = len(a)
    return sum([(a[i] - b[i]) ** 2 for i in range(n)])

def sqr_distance_point_to_centroid(data, centroids, i, k):
    # d(data[i].c1, centr[k].c1) ** 2
    dc1_sq = sqr_distance([data[i][j].c1 for j in range(dim)], [centroids[k][j].c1 for j in range(dim)])
    # d(data[i].c2, centr[k].c2) ** 2
    dc2_sq = sqr_distance([data[i][j].c2 for j in range(dim)], [centroids[k][j].c2 for j in range(dim)])
    # d(data[i].l, centr[k].l) ** 2
    dl_sq = sqr_distance([data[i][j].l for j in range(dim)], [centroids[k][j].l for j in range(dim)])
    # d(data[i].r, centr[k].r) ** 2
    dr_sq = sqr_distance([data[i][j].r for j in range(dim)], [centroids[k][j].r for j in range(dim)])
    return dc1_sq, dc2_sq, dl_sq, dr_sq

def cmeans_fuzzy_data(data, c, m, error, maxiter):
    """
    Кластеризация нечетких данных методом Fuzzy c-means
    (см. doi.org/10.1016/j.csda.2010.09.013)

    data: list(list(FuzzyNumber)), size N_samples, N_features
    c: int
        Число кластеров
    m: float
        Степенной параметр фаззификации
    error: float
        Допустимая разница двух последовательных приближений (в условии остановки)
    maxiter: int
        Максимальное число итераций

    Возвращает:
        centers, membership_degrees

        centers: list(list(FuzzyNumber)), size c, N_features
            Координаты центров кластеров
        membership_degrees: list(list(float)), size N_samples, c
    """

    # размер выборки
    data_size = len(data)
    # размерность пространства признаков
    dim = len(data[0])

    # центроиды - массив нечетких чисел размерности c, dim
    centroids = [[FuzzyNumber(0, 0, 0, 0)] * dim for _ in range(c)]

    # меры принадлежности точек к кластерам - массив чисел от 0 до 1 размерности data_size, c
    #   сумма мер принадлежности по всем кластерам равна 1
    u = [[] for _ in range(data_size)]
    for i in range(data_size):
        # генерируем случайные меры принадлежности u[i] и считаем их сумму s
        s = 0
        for j in range(c):
            x = random.random()
            u[i].append(x)
            s += x
        # обеспечиваем условие: сумма мер принадлежности равна 1
        for j in range(c):
            u[i][j] /= s

    for iter in range(maxiter):

        # вычисляем нечеткие центроиды (из условия оптимальности функционала качества)
        for k in range(c):
            # считаем сумму m-х степеней мер принадлежности по всем точкам выборки
            s = 0
            for i in range(data_size):
                s += u[i][k] ** m
            for j in range(dim):
                # усредняем data[i][j] с весами u[i][k] ** m
                centroids[k][j] = FuzzyNumber(0, 0, 0, 0)
                for i in range(data_size):
                    centroids[k][j].c1 += (u[i][k] ** m) * data[i][j].c1
                    centroids[k][j].c2 += (u[i][k] ** m) * data[i][j].c2
                    centroids[k][j].l += (u[i][k] ** m) * data[i][j].l
                    centroids[k][j].r += (u[i][k] ** m) * data[i][j].r
                centroids[k][j].c1 /= s
                centroids[k][j].c2 /= s
                centroids[k][j].l /= s
                centroids[k][j].r /= s

        # вычисляем весовые коэффициенты wc и ws
        numerator = 0
        denominator = 0
        for i in range(data_size):
            for k in range(c):
                dc1_sq, dc2_sq, dl_sq, dr_sq = sqr_distance_point_to_centroid(data, centroids, i, k)
                term1 = (u[i][k] ** m) * (dl_sq + dr_sq)
                numerator += term1
                term2 = (u[i][k] ** m) * (dc1_sq + dc2_sq + dl_sq + dr_sq)
                denominator += term2
        wc = max(numerator / denominator, 0.5)
        ws = 1 - wc

        # вычисляем меры принадлежности точек к кластерам
        new_u = [[None] * c for _ in range(data_size)]
        for i in range(data_size):
            s = 0
            for k in range(c):
                dc1_sq, dc2_sq, dl_sq, dr_sq = sqr_distance_point_to_centroid(data, centroids, i, k)
                s += ( (wc ** 2) * (dc1_sq + dc2_sq) + (ws ** 2) * (dl_sq + dr_sq) ) ** (-1 / (m - 1))
            for k in range(c):
                dc1_sq, dc2_sq, dl_sq, dr_sq = sqr_distance_point_to_centroid(data, centroids, i, k)
                term = ( (wc ** 2) * (dc1_sq + dc2_sq) + (ws ** 2) * (dl_sq + dr_sq) ) ** (-1 / (m - 1))
                new_u[i][k] = term / s

        # обновляем меры принадлежности
        for i in range(data_size):
            for k in range(c):
                u[i][k] = new_u[i][k]

    return centroids, u


def fuzzify(val):
    """
    Возвращает нечеткое число, соответствующее четкому значению val
    """
    if val == 0:
        c1 = val
        c2 = val
        l = 0
        r = 1
    elif val == 5:
        c1 = val
        c2 = val
        l = 1
        r = 0
    else:
        c1 = val
        c2 = val
        l = 1
        r = 1
    return FuzzyNumber(c1, c2, l, r)


dim = 10  # размерность пространства признаков

# чтение входных данных
with open("data.csv") as fp:
    reader = csv.reader(fp, delimiter=";")
    next(reader, None)  # пропустить заголовки
    data_str = [row for row in reader]

# преобразование данных в числовой формат
data_size = len(data_str)  # число точек для кластеризации
data = [[] for _ in range(data_size)]
for i in range(data_size):
    data[i] = list(map(int, data_str[i]))
    assert(len(data[i]) == dim)

# фаззификация данных
fdata = [[] for _ in range(data_size)]
for i in range(data_size):
    for j in range(dim):
        fdata[i].append(fuzzify(data[i][j]))

# кластеризация
clust_c = 3  # число кластеров
clust_m = 1.3  # степенной параметр фаззификации
clust_error = 1e-5
clust_maxiter = 1000
centers, membership_degrees = cmeans_fuzzy_data(fdata, clust_c, clust_m, clust_error, clust_maxiter)

print("Центроиды:\n", centers)
print("Меры принадлежности точек к кластерам:\n", membership_degrees)

with open("clustering_result.txt", "wt") as fp:
    fp.write('Центроиды:\n')
    for cntr in centers:
        for fuz_number in cntr:
            fp.write(str(fuz_number))
            fp.write('; ')
        fp.write('\n')
    fp.write('\n\nМеры принадлежности точек к кластерам:\n')
    for sample_point in membership_degrees:
        for x in sample_point:
            fp.write(str(x))
            fp.write(" ")
        fp.write('\n')

    threshold = 0.6
    clusters_points = [[] for _ in range(clust_c)]
    outliers = []
    for point_index, point_memberships in enumerate(membership_degrees):
        best_cluster = None
        for cluster_index, mmbr in enumerate(point_memberships):
            if mmbr > threshold:
                best_cluster = cluster_index
        if best_cluster is not None:
            clusters_points[best_cluster].append(point_index)
        else:
            outliers.append(point_index)
    
    fp.write('\n\n\n')
    for cluster_index in range(clust_c):
        fp.write('Кластер ')
        fp.write(str(cluster_index + 1))
        fp.write(" (")
        for fuz_number in centers[cluster_index]:
            x = ((fuz_number.c1 - fuz_number.l) + (fuz_number.c2 + fuz_number.r)) / 2
            fp.write('%.2f' % x)
            fp.write(", ")
        fp.write(")")
        fp.write(':\n')
        for point_index in clusters_points[cluster_index]:
            for x in membership_degrees[point_index]:
                fp.write(str(x))
                fp.write(" ")
            fp.write("  (")
            for x in data[point_index]:
                fp.write(str(x))
                fp.write(" ")
            fp.write(")")
            fp.write(" [" + str(point_index + 1) + "]")
            fp.write("\n")
        fp.write("\n")
    fp.write("Остальные точки:\n")
    for point_index in outliers:
        for x in membership_degrees[point_index]:
            fp.write(str(x))
            fp.write(" ")
        fp.write("  (")
        for x in data[point_index]:
            fp.write(str(x))
            fp.write(" ")
        fp.write(")")
        fp.write(" [" + str(point_index + 1) + "]")
        fp.write("\n")

    fp.write("\n\nТочки к кластерам:\n")
    for cluster_index in range(clust_c):
        for point_index in clusters_points[cluster_index]:
            fp.write(str(point_index + 1) + " ")
        fp.write("\n")
