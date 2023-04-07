n = 3  # число кластеров по первой кластеризации
m = 3  # число кластеров по второй кластеризации

def read_clust(filename, n):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    assert(len(lines) == n)
    clust = []
    for line in lines:
        clust.append(list(map(int, line.split(" "))))
    return clust

# сколько (в %) точек из списка cluster1 содержатся в списке cluster2
def calc_percentage(cluster1, cluster2):
    counter = 0
    for elem1 in cluster1:
        if elem1 in cluster2:
            counter += 1
    return counter / len(cluster1) * 100

def make_table(clust1, clust2):
    n = len(clust1)
    m = len(clust2)
    table = [[0] * m for _ in range(n)]
    for i, cluster1 in enumerate(clust1):
        for j, cluster2 in enumerate(clust2):
            # считаем, сколько точек кластера clust1 попали в кластер clust2
            table[i][j] = calc_percentage(cluster1, cluster2)
    return table

clust1 = read_clust("clust1.txt", n)
clust2 = read_clust("clust2.txt", m)

print(make_table(clust1, clust2))
print(make_table(clust2, clust1))
