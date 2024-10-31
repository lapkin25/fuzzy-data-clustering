import numpy as np
import matplotlib.pyplot as plt


def read_list(f):
    strs = f.readline().strip().split()
    if strs[0][0] == '[':
        strs[0] = strs[0][1:]
        if len(strs[0]) == 0:
            strs = strs[1:]
    z = list(map(float, strs))
    finish = False
    while not finish:
        strs = f.readline().strip().split()
        if strs[-1][-1] == ']':
            finish = True
            strs[-1] = strs[-1][:-1]
            if len(strs[-1]) == 0:
                strs = strs[:-1]
        z += list(map(float, strs))
    return z


# индексы направлений мероприятий, относящихся к каждому блоку
activities_blocks = [[0, 1, 2],
                     [3, 4, 5, 6],
                     [7, 8],
                     [9, 10, 11, 12, 13],
                     [14, 15, 16, 17, 18],
                     [19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]

with open("Случайные реализации/Сдвиг границ_за год/10_6_год.txt", 'r') as f:
    s = f.readline().strip()
    s = f.readline().strip()
    kpi_vec = np.zeros(100)
    z_vec = np.zeros((4, 100, 29))
    z_blocks_vec = np.zeros((4, 100, 6))
    for i in range(100):
        s = f.readline().strip()
        kpi = float(s)
        print(kpi)
        kpi_vec[i] = kpi

        for t in range(4):
            z = read_list(f)
            print(z)
            assert(len(z) == 29)
            z_vec[t, i, :] = z

            for k, block in enumerate(activities_blocks):
                z_blocks_vec[t, i, k] = sum([z_vec[t, i, j] for j in block])

print("mu = ", np.mean(kpi_vec))
print("sigma = ", np.std(kpi_vec))
mu_Z = np.mean(z_vec, axis=1)
print("mu_Z = ", np.mean(z_vec, axis=1))
print("sigma_Z = ", np.std(z_vec, axis=1))

mu_blocks = np.mean(z_blocks_vec, axis=1)
sigma_blocks = np.std(z_blocks_vec, axis=1)
print("mu_blocks = ", mu_blocks)
print("sigma_blocks = ", sigma_blocks)

z_blocks_year = np.sum(z_blocks_vec, axis=0)

mu_blocks_year = np.mean(z_blocks_year, axis=0)
sigma_blocks_year = np.std(z_blocks_year, axis=0)
print("mu_blocks_year = ", mu_blocks_year)
print("sigma_blocks_year = ", sigma_blocks_year)

