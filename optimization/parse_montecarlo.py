import numpy as np

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


with open("Случайные реализации/5_3.txt", 'r') as f:
    s = f.readline().strip()
    s = f.readline().strip()
    kpi_vec = np.zeros(100)
    z_vec = np.zeros((100, 29))
    c_vec = np.zeros((100, 6))
    for i in range(100):
        s = f.readline().strip()
        kpi = float(s)
        print(kpi)
        kpi_vec[i] = kpi

        z = read_list(f)
        print(z)
        assert(len(z) == 29)
        z_vec[i, :] = z

        c = read_list(f)
        print(c)
        assert(len(c) == 6)
        c_vec[i, :] = c

print("mu = ", np.mean(kpi_vec))
print("sigma = ", np.std(kpi_vec))
print("mu_Z = ", np.mean(z_vec, axis=0))
print("sigma_Z = ", np.std(z_vec, axis=0))
print("mu_classes = ", np.mean(c_vec, axis=0))
print("sigma_classes = ", np.std(c_vec, axis=0))
