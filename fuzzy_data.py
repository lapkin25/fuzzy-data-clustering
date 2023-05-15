import csv
import random
import math
from progress.bar import Bar


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
        return "(c1 = %.2f, c2 = %.2f, l = %.2f, r = %.2f)" % (self.c1, self.c2, self.l, self.r)

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
            степени принадлежности точек к кластерам
        wc, ws: float
            коэффициенты в расстоянии между нечеткими числами
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

    bar = Bar('Iterations', max=maxiter)

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
                s += ((wc ** 2) * (dc1_sq + dc2_sq) + (ws ** 2) * (dl_sq + dr_sq)) ** (-1 / (m - 1))
            for k in range(c):
                dc1_sq, dc2_sq, dl_sq, dr_sq = sqr_distance_point_to_centroid(data, centroids, i, k)
                term = ((wc ** 2) * (dc1_sq + dc2_sq) + (ws ** 2) * (dl_sq + dr_sq)) ** (-1 / (m - 1))
                new_u[i][k] = term / s

        # обновляем меры принадлежности
        for i in range(data_size):
            for k in range(c):
                u[i][k] = new_u[i][k]
        bar.next()
    bar.finish()
    return centroids, u, wc, ws


def fuzzy_distance(a, b, wc, ws):
    """
    Возвращает расстояние между нечеткими числами a и b.
    """
    # d(data[i].c1, centr[k].c1) ** 2
    dc1_sq = sqr_distance([a[j].c1 for j in range(dim)], [b[j].c1 for j in range(dim)])
    # d(data[i].c2, centr[k].c2) ** 2
    dc2_sq = sqr_distance([a[j].c2 for j in range(dim)], [b[j].c2 for j in range(dim)])
    # d(data[i].l, centr[k].l) ** 2
    dl_sq = sqr_distance([a[j].l for j in range(dim)], [b[j].l for j in range(dim)])
    # d(data[i].r, centr[k].r) ** 2
    dr_sq = sqr_distance([a[j].r for j in range(dim)], [b[j].r for j in range(dim)])
    return math.sqrt((wc ** 2) * (dc1_sq + dc2_sq) + (ws ** 2) * (dl_sq + dr_sq))


def fuzzy_distance_component(a, b, wc, ws, i):
    """
    Возвращает i-ю компоненту расстояния между нечеткими числами a и b.
    """
    dc1_sq = (a[i].c1 - b[i].c1) ** 2
    dc2_sq = (a[i].c2 - b[i].c2) ** 2
    dl_sq = (a[i].l - b[i].l) ** 2
    dr_sq = (a[i].r - b[i].r) ** 2
    return math.sqrt((wc ** 2) * (dc1_sq + dc2_sq) + (ws ** 2) * (dl_sq + dr_sq))


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
    assert (len(data[i]) == dim)

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
centers, membership_degrees, wc, ws = cmeans_fuzzy_data(fdata, clust_c, clust_m, clust_error, clust_maxiter)

#print("Центроиды:\n", centers)
#print("Меры принадлежности точек к кластерам:\n", membership_degrees)

with open("clustering_result.txt", "wt") as fp:
    fp.write('wc = %.2f, ws = %.2f\n' % (ws, wc))
    for cluster_index in range(clust_c):
        fp.write('Кластер ')
        fp.write(str(cluster_index + 1))
        fp.write(" (")
        for fuz_number in centers[cluster_index]:
            x = (fuz_number.c1 + fuz_number.c2) / 2 + (fuz_number.r - fuz_number.l) / 4
            fp.write('%.2f' % x)
            fp.write(", ")
        fp.write(")\n")
    fp.write('Центроиды:\n')
    for cntr in centers:
        for fuz_number in cntr:
            fp.write(str(fuz_number))
            fp.write('; ')
        fp.write('\n')

#    dist_centr = [[] for _ in range(clust_c)]  # расстояния между центроидами
#    for i in range(clust_c):
#        for j in range(clust_c):
#            dist_centr[i].append(fuzzy_distance(centers[i], centers[j], wc, ws))
#    fp.write('\nРасстояния между центроидами:\n')
#    for i in range(clust_c):
#        for j in range(clust_c):
#            fp.write("%.2f " % dist_centr[i][j])
#        fp.write("\n")

    with open("u.txt", "wt") as fu:
        #fp.write('\n\nМеры принадлежности точек к кластерам:\n')
        for sample_point in membership_degrees:
            for x in sample_point:
                fu.write("%.2f " % x)
            fu.write('\n')
    with open("d.txt", "wt") as fd:
        #fp.write('\n\nРасстояния от точек до центроидов:\n')
        for i in range(data_size):
            for j in range(clust_c):
                fd.write("%.2f " % fuzzy_distance(fdata[i], centers[j], wc, ws))
            fd.write('\n')

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
            x = (fuz_number.c1 + fuz_number.c2) / 2 + (fuz_number.r - fuz_number.l) / 4
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
            fp.write(" d = %.2f" % fuzzy_distance(fdata[point_index], centers[cluster_index], wc, ws))
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

    #fp.write("\n\n")
    #for i in range(dim):
    #    fp.write("Признак %d: доверительные интервалы для центроидов:\n" % (i + 1))
    #    for cluster_index in range(clust_c):
    #        rms = 0
    #        for point_index in range(data_size):
    #            rms += membership_degrees[point_index][cluster_index] * fuzzy_distance_component(
    #                fdata[point_index], centers[cluster_index], wc, ws, i) ** 2
    #        rms = math.sqrt(rms / data_size)
    #        delta = rms / math.sqrt(sum([membership_degrees[point_index][cluster_index] for point_index in range(data_size)])) * 1.64  # 10%-ный доверительный интервал
    #        fp.write("+/-%.2f  " % delta)
    #    fp.write("\n")

    fp.write("\n\n")
    for j in range(dim):
        fp.write("Компонента %d: доверительный интервал для среднего\n" % (j + 1))
        for k in range(clust_c):
            # считаем среднее и дисперсию с учетом мер принадлежности по всем точкам выборки
            s = 0
            for i in range(data_size):
                s += membership_degrees[i][k]
            # усредняем fdata[i][j] с весами u[i][k]
            cluster_mean_j = FuzzyNumber(0, 0, 0, 0)
            for i in range(data_size):
                cluster_mean_j.c1 += membership_degrees[i][k] * fdata[i][j].c1
                cluster_mean_j.c2 += membership_degrees[i][k] * fdata[i][j].c2
                cluster_mean_j.l += membership_degrees[i][k] * fdata[i][j].l
                cluster_mean_j.r += membership_degrees[i][k] * fdata[i][j].r
            cluster_mean_j.c1 /= s
            cluster_mean_j.c2 /= s
            cluster_mean_j.l /= s
            cluster_mean_j.r /= s

            cluster_variance_j = FuzzyNumber(0, 0, 0, 0)
            for i in range(data_size):
                cluster_variance_j.c1 += membership_degrees[i][k] * (fdata[i][j].c1 - cluster_mean_j.c1) ** 2
                cluster_variance_j.c2 += membership_degrees[i][k] * (fdata[i][j].c2 - cluster_mean_j.c2) ** 2
                cluster_variance_j.l += membership_degrees[i][k] * (fdata[i][j].l - cluster_mean_j.l) ** 2
                cluster_variance_j.r += membership_degrees[i][k] * (fdata[i][j].r - cluster_mean_j.r) ** 2
            cluster_variance_j.c1 /= s
            cluster_variance_j.c2 /= s
            cluster_variance_j.l /= s
            cluster_variance_j.r /= s

            delta = FuzzyNumber(
                math.sqrt(cluster_variance_j.c1) / math.sqrt(s) * 1.64,
                math.sqrt(cluster_variance_j.c2) / math.sqrt(s) * 1.64,
                math.sqrt(cluster_variance_j.l) / math.sqrt(s) * 1.64,
                math.sqrt(cluster_variance_j.r) / math.sqrt(s) * 1.64)

            confidence_left = FuzzyNumber(
                cluster_mean_j.c1 - delta.c1,
                cluster_mean_j.c2 - delta.c2,
                cluster_mean_j.l - delta.l,
                cluster_mean_j.r - delta.r)
            confidence_right = FuzzyNumber(
                cluster_mean_j.c1 + delta.c1,
                cluster_mean_j.c2 + delta.c2,
                cluster_mean_j.l + delta.l,
                cluster_mean_j.r + delta.r)
            median_confidence_left = (confidence_left.c1 + confidence_left.c2) / 2 + (confidence_left.r - confidence_right.l) / 4
            median_confidence_right = (confidence_right.c1 + confidence_right.c2) / 2 + (confidence_right.r - confidence_left.l) / 4
            
            fp.write("  Кластер %d :  (%.2f - %.2f)\n" % (k + 1, median_confidence_left, median_confidence_right))
                

            #fp.write("    c1 = (%.2f - %.2f)  " % (cluster_mean_j.c1 - delta.c1, cluster_mean_j.c1 + delta.c1))
            #fp.write("    c2 = (%.2f - %.2f)  " % (cluster_mean_j.c2 - delta.c2, cluster_mean_j.c2 + delta.c2))
            #fp.write("    l  = (%.2f - %.2f)  " % (cluster_mean_j.l - delta.l, cluster_mean_j.l + delta.l))
            #fp.write("    r  = (%.2f - %.2f)  " % (cluster_mean_j.r - delta.r, cluster_mean_j.r + delta.r))
            #fp.write("\n")



    # fp.write("\n\n")
    # for i in range(dim):
    #     fp.write("Компонента %d: разбросы в пределах каждого кластера\n" % (i + 1))
    #     for cluster_index in range(clust_c):
    #         rms = 0
    #         for point_index in range(data_size):
    #             rms += membership_degrees[point_index][cluster_index] * fuzzy_distance_component(
    #                 fdata[point_index], centers[cluster_index], wc, ws, i) ** 2
    #         rms = math.sqrt(rms / data_size)
    #         fp.write("%.2f  " % rms)
    #     fp.write("\n")
    #
    # fp.write("\n\n")
    # for i in range(dim):
    #     fp.write("Компонента %d: расстояния между центроидами по этой компоненте\n" % (i + 1))
    #     for k in range(clust_c):
    #         for l in range(clust_c):
    #             fp.write("%.2f " % fuzzy_distance_component(centers[k], centers[l], wc, ws, i))
    #         fp.write("\n")
    #
    # spreads = []  # среднеквадратичные разбросы внутри каждого кластера
    # for cluster in range(clust_c):
    #     rms = 0
    #     for point in range(data_size):
    #         rms += membership_degrees[point][cluster] * fuzzy_distance(fdata[point], centers[cluster], wc, ws) ** 2
    #     rms = math.sqrt(rms / data_size)
    #     spreads.append(rms)
    # fp.write("\nРазбросы внутри кластеров:\n")
    # for i in range(clust_c):
    #     fp.write("%.2f  " % spreads[i])
