n = 3  # число кластеров по первой кластеризации
m = 3  # число кластеров по второй кластеризации

data_size = 520

def read_floats (file_name, num_rows, num_cols):
    a = []
    with open(file_name) as f:
        for line in f:
            a.append(list(map(float, line.split())))
            assert (len(a[-1]) == num_cols)
    assert (len(a) == num_rows)
    return a


u1 = read_floats("u1.txt", data_size, n)
u2 = read_floats("u2.txt", data_size, m)

def make_table(u1, u2):
    p = []
    for i in range(n):
        p.append([])
        for j in range(m):
            # считаем, сколько % точек i-го кластера (по первой кластеризации) принадлежат j-му кластеру (по второй кластеризации)
            sum1 = 0
            sum2 = 0
            for k in range(data_size):
                sum1 += u1[k][i] * u2[k][j]
                sum2 += u1[k][i]
            p[i].append(sum1 / sum2)
    return p

def print_table(table):
    with open("compare.txt", "wt") as fp:
        for i in range(n):
            for j in range(m):
                fp.write('%.2f  ' % table[i][j])
            fp.write("\n")


print_table(make_table(u1, u2))
print_table(make_table(u2, u1))
