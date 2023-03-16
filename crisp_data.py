import csv
import skfuzzy as fuzz
import numpy as np

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

npdata = np.transpose(np.array(data))
clust_c = 5  # число кластеров
clust_m = 1.5  # степенной параметр фаззификации
clust_error = 1e-5
clust_maxiter = 1000
centers, membership_degrees, _, _, _, _, _ = fuzz.cmeans(npdata, clust_c, clust_m, clust_error, clust_maxiter)

print("scikit-fuzzy")
print("Центры кластеров:\n", centers)
print("Меры принадлежности точек к кластерам:\n", np.transpose(membership_degrees))

from fcmeans import FCM

npdata = np.array(data)
fcm = FCM(n_clusters=clust_c, m=clust_m)
fcm.fit(npdata)
centers = fcm.centers
membership_degrees = fcm.soft_predict(npdata)

print("\n\nfuzzy-c-means")
print("Центры кластеров:\n", centers)
print("Меры принадлежности точек к кластерам:\n", membership_degrees)
