import math
import matplotlib.pyplot as plt

num_clusters = 3
data_size = 520

def read_floats (file_name, num_rows, num_cols):
    a = []
    with open(file_name) as f:
        for line in f:
            a.append(list(map(float, line.split())))
            assert (len(a[-1]) == num_cols)
    assert (len(a) == num_rows)
    return a


u = read_floats("u.txt", data_size, num_clusters)
d = read_floats("d.txt", data_size, num_clusters)

spreads = []  # среднеквадратичные разбросы внутри каждого кластера
for cluster in range(num_clusters):
    rms = 0
    for point in range(data_size):
        rms += u[point][cluster] * d[point][cluster] ** 2
    rms = math.sqrt(rms / data_size)
    spreads.append(rms)

print("Разбросы внутри кластеров:", spreads)

bars = [[] for _ in range(num_clusters)]
for cluster in range(num_clusters):
    max_distance = math.ceil(max([d[point][cluster] for point in range(data_size)]))
    bars[cluster] = [0] * max_distance
    for dist in range(max_distance):
        # считаем сумму мер принадлежности точек с расстояниями d, dist <= d < dist + 1
        for point in range(data_size):
            if dist <= d[point][cluster] and d[point][cluster] < dist + 1:
                bars[cluster][dist] += u[point][cluster]

# построение столбчатых диаграмм
print(bars)
fig, axes = plt.subplots(num_clusters, 1)
for cluster in range(num_clusters):
    x = range(len(bars[cluster]))
    y = bars[cluster]
    axes[cluster].bar(x, y)
plt.show()
