n = 3  # число кластеров

data_size = 139

def read_floats (file_name, num_rows, num_cols):
    a = []
    with open(file_name) as f:
        for line in f:
            a.append(list(map(float, line.split())))
            assert (len(a[-1]) == num_cols)
    assert (len(a) == num_rows)
    return a


u = read_floats("u1.txt", data_size, n)
s = []
for i in range(n):
    s.append(sum([u[k][i] for k in range(data_size)]))
print(s)

with open("count.txt", "wt") as fp:
    for threshold in [0.6, 0.7, 0.8, 0.9]:
        cnt = [0] * n
        for k in range(data_size):
            for i in range(n):
                if u[k][i] > threshold:
                    cnt[i] += 1
        # cnt[i] - количество точек с мерой принадлежности i-му кластеру выше threshold
        fp.write(str(threshold))
        fp.write("  ")
        for i in range(n):
            fp.write(str(cnt[i]))
            fp.write(" ")
        fp.write("\n")

with open("points.txt", "wt") as fp:
    threshold = 0.7
    points_to_cluster = [[] for _ in range(n)]
    for k in range(data_size):
        for i in range(n):
            if u[k][i] > threshold:
                points_to_cluster[i].append(k + 1)
    fp.write("Точки, отнесенные к кластерам:\n")
    for i in range(n):
        fp.write("Кластер %d:\n" % (i + 1))
        for p in points_to_cluster[i]:
            fp.write("%d " % p)
        fp.write("\n")
