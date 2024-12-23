from input_data import *
from optimize_kpi import optimize, calc_kpi


# исходные данные
expectations = ExpectationsData("data_deviation_expectations.csv")
compet_t0 = CompetData("data_compet_t0.csv")
burnout_t0 = BurnoutData("data_burnout_t0.csv")
kpi_t0 = KPIData("data_kpi_t0.csv")

# эконометрические зависимости
invest_to_compet = InvestToCompet("invest_to_compet.csv")
activities_expectations = ActivitiesExpectations("data_min_max_expectations.csv")
expectations_to_burnout = ExpectationsToBurnout(expectations)
compet_burnout_to_kpi = CompetBurnoutToKPI(compet_t0)

# ограничения оптимизации
budget_constraints = BudgetConstraints("budget_activities.csv")

# создаем случайную выборку из 100 людей
selected_data_size = 100
random_state = 1
np.random.seed(random_state)
indices = np.random.permutation(data_size)
selected = indices[:selected_data_size]
print("Индексы выбранных сотрудников: ", selected)

# бюджет: 500000 в год на человека
total_budget = 500000 / 4 * selected_data_size

# индексы направлений мероприятий, относящихся к каждому блоку
activities_blocks = [[0, 1, 2],
                     [3, 4, 5, 6],
                     [7, 8],
                     [9, 10, 11, 12, 13],
                     [14, 15, 16, 17, 18],
                     [19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]

block_name = ['Финансовое благополучие', 'Окружающая среда', 'Карьерное развитие',
              'Здоровый образ жизни', 'Развитие навыков', 'Корпоративная инфраструктура']

# [a, b] - это доверительный интервал mu ± 2 * sigma
def trunc_normal_random(a, b):
    ok = False
    while not ok:
        x = np.random.normal((a + b) / 2, (b - a) / 4)
        if x >= a and x <= b:
            ok = True
    #print(a, b, x)
    return x


def uniform_random(a, b):
    return np.random.uniform(a, b)


# Найти возмущенный коэффициент с относительным изменением delta
def perturb_coef(coef, delta):
    sigma = abs(coef) * delta
    return trunc_normal_random(coef - sigma, coef + sigma)


# Найти возмущенный коэффициент (с равномерным распределением) с относительным увеличением delta
def perturb_coef_uniform(coef, rel_shift):
    delta = abs(coef) * rel_shift
    return uniform_random(coef, coef + delta)


# Варьируем коэффициенты модели, разыгрывая их на доверительных интервалах
# Находим случайную реализацию оптимального KPI
# delta - относительное возмущение коэффициентов, epsilon - относительное возмущение ломаной
# Возвращает кортеж: оптимальный KPI, распредедение инвестиций по направлениям
def generate_random_kpi(delta, epsilon):
    # Разыгрываем коэффициенты влияния инвестиций на компетенции
    perturbed_invest_to_compet = InvestToCompet("invest_to_compet.csv")
    # просто присвоить один объект другому нельзя, поэтому создали еще один такой же
    for j in range(num_compet):
        #print("Проверка:", invest_to_compet.beta[j], "[", invest_to_compet_left_conf.beta[j], ",",
        #      invest_to_compet_right_conf.beta[j], "]")
        for k in range(num_activities):
            #print("Проверка:", invest_to_compet.alpha[j, k], "[", invest_to_compet_left_conf.alpha[j, k], ",",
            #      invest_to_compet_right_conf.alpha[j, k], "]")
            perturbed_invest_to_compet.beta[j] =\
                trunc_normal_random(invest_to_compet_left_conf.beta[j], invest_to_compet_right_conf.beta[j])
            perturbed_invest_to_compet.alpha[j, k] =\
                trunc_normal_random(invest_to_compet_left_conf.alpha[j, k], invest_to_compet_right_conf.alpha[j, k])

    # Разыгрываем коэффициенты зависимости KPI от компетенций
    perturbed_compet_burnout_to_kpi = CompetBurnoutToKPI(compet_t0)
    for m in range(num_kpi_indicators):
        for j in range(num_compet):
            perturbed_compet_burnout_to_kpi.w[m, j] = perturb_coef(compet_burnout_to_kpi.w[m, j], delta)

    # Разыгрываем коэффициенты зависимости KPI от выгорания
    for m in range(num_kpi_indicators):
        perturbed_compet_burnout_to_kpi.burnout_intercept[m] =\
            perturb_coef(compet_burnout_to_kpi.burnout_intercept[m], delta)
        for l in range(num_burnout_indicators):
            perturbed_compet_burnout_to_kpi.burnout_coef[m, l] =\
                perturb_coef(compet_burnout_to_kpi.burnout_coef[m, l], delta)

    # Разыгрываем коэффициенты зависимости выгорания от ожиданий
    perturbed_expectations_to_burnout = ExpectationsToBurnout(expectations)
    for l in range(num_burnout_indicators):
        for k in range(num_activities):
            perturbed_expectations_to_burnout.w[l, k] = perturb_coef(expectations_to_burnout.w[l, k], delta)

    # Разыгрываем кусочно-линейную зависимость KPI от компетенций
    for m in range(num_kpi_indicators):
        for p in range(num_compet_classes):
            perturbed_compet_burnout_to_kpi.c[m, p] = perturb_coef(compet_burnout_to_kpi.c[m, p], epsilon)
        ok = False
        while not ok:
            for p in range(num_compet_classes):
                perturbed_compet_burnout_to_kpi.r[m, p] = perturb_coef(compet_burnout_to_kpi.r[m, p], epsilon)
            ok = True
            for p in range(num_compet_classes - 1):
                if perturbed_compet_burnout_to_kpi.r[m, p] > perturbed_compet_burnout_to_kpi.r[m, p + 1]:
                    ok = False

    # Разыгрываем кусочно-линейную зависимость выгорания от ожиданий
    for l in range(num_burnout_indicators):
        for p in range(num_expectation_classes):
            perturbed_expectations_to_burnout.e[l, p] = perturb_coef(expectations_to_burnout.e[l, p], epsilon)
        ok = False
        while not ok:
            for p in range(num_expectation_classes):
                perturbed_expectations_to_burnout.r[l, p] = perturb_coef(expectations_to_burnout.r[l, p], epsilon)
            ok = True
            for p in range(num_expectation_classes - 1):
                if perturbed_expectations_to_burnout.r[l, p] > perturbed_expectations_to_burnout.r[l, p + 1]:
                    ok = False

    z, x_new, q_new = optimize(compet_t0.x[selected, :], expectations.q[selected, :], expectations.a[selected, :],
                               perturbed_invest_to_compet, activities_expectations, perturbed_expectations_to_burnout,
                               perturbed_compet_burnout_to_kpi, budget_constraints, total_budget, verbose=False)

    kpi1 = calc_kpi(x_new, q_new, expectations.a[selected, :],
                    perturbed_expectations_to_burnout, perturbed_compet_burnout_to_kpi)
    integral_kpi = np.dot(np.mean(kpi1, axis=0), compet_burnout_to_kpi.kpi_importance)

    #print("Распределение по направлениям:", np.sum(z, axis=0))

    # распределение инвестиций по компетентностным классам
    invest_by_compet_classes = np.zeros(num_compet_classes)
    for j, i in enumerate(selected):
        u = compet_burnout_to_kpi.calc_u(2, compet_t0.x[i, :])  # берем категории для KPI с индексом 2
        #print(u)
        for k in range(num_compet_classes):
            invest_by_compet_classes[k] += u[k] * np.sum(z[j, :])

    return integral_kpi, np.sum(z, axis=0), invest_by_compet_classes


# Найти среднеквадратичный разброс при параметрах delta, epsilon с num_samples случайных реализаций
def calc_mean_std(num_samples, delta, epsilon, file):
    fout_kpi = open(file, 'w')
    print("delta =", delta, ", epsilon =", epsilon, "\n", file=fout_kpi)
    kpi_sample = np.zeros(num_samples)
    Z = np.zeros((num_samples, num_activities))
    invest_by_compet_classes = np.zeros((num_samples, num_compet_classes))
    for i in range(num_samples):
        print("\n", "РЕАЛИЗАЦИЯ", i + 1, "\n")
        kpi_sample[i], Z[i, :], invest_by_compet_classes[i, :] = generate_random_kpi(delta, epsilon)
        print(kpi_sample[i], file=fout_kpi)
        print(Z[i, :], file=fout_kpi)
        print(invest_by_compet_classes[i, :], file=fout_kpi)
    mu = np.mean(kpi_sample)
    sigma = np.std(kpi_sample)
    mu_Z = np.mean(Z, axis=0)
    sigma_Z = np.std(Z, axis=0)
    mu_classes = np.mean(invest_by_compet_classes, axis=0)
    sigma_classes = np.std(invest_by_compet_classes, axis=0)
    print("\n", "mu =", mu, ", sigma =", sigma, file=fout_kpi)
    print("mu_Z =", mu_Z, file=fout_kpi)
    print("sigma_Z =", sigma_Z, file=fout_kpi)
    print("mu_classes =", mu_classes, file=fout_kpi)
    print("sigma_classes =", sigma_classes, file=fout_kpi)
    fout_kpi.close()
    return mu, sigma, mu_Z, sigma_Z, mu_classes, sigma_classes


def generate_random_kpi_constr(rel_shift, block_index):
    perturbed_budget_constraints = BudgetConstraints("budget_activities.csv")
    # сдвигаем границы для заданного блока
    for ind in activities_blocks[block_index]:
        perturbed_budget_constraints.budget_activities_percent[ind] =\
            perturb_coef_uniform(perturbed_budget_constraints.budget_activities_percent[ind], rel_shift)

    z, x_new, q_new = optimize(compet_t0.x[selected, :], expectations.q[selected, :], expectations.a[selected, :],
                               invest_to_compet, activities_expectations, expectations_to_burnout,
                               compet_burnout_to_kpi,
                               perturbed_budget_constraints, total_budget, verbose=False)

    kpi1 = calc_kpi(x_new, q_new, expectations.a[selected, :],
                    expectations_to_burnout, compet_burnout_to_kpi)
    integral_kpi = np.dot(np.mean(kpi1, axis=0), compet_burnout_to_kpi.kpi_importance)

    # распределение инвестиций по компетентностным классам
    invest_by_compet_classes = np.zeros(num_compet_classes)
    for j, i in enumerate(selected):
        u = compet_burnout_to_kpi.calc_u(2, compet_t0.x[i, :])  # берем категории для KPI с индексом 2
        #print(u)
        for k in range(num_compet_classes):
            invest_by_compet_classes[k] += u[k] * np.sum(z[j, :])

    return integral_kpi, np.sum(z, axis=0), invest_by_compet_classes


def generate_random_kpi_constr_year(rel_shift, block_index):
    perturbed_budget_constraints = BudgetConstraints("budget_activities.csv")
    # сдвигаем границы для заданного блока
    for ind in activities_blocks[block_index]:
        perturbed_budget_constraints.budget_activities_percent[ind] =\
            perturb_coef_uniform(perturbed_budget_constraints.budget_activities_percent[ind], rel_shift)

    print("квартал", 1)
    z, x_new, q_new = optimize(compet_t0.x[selected, :], expectations.q[selected, :], expectations.a[selected, :],
                               invest_to_compet, activities_expectations, expectations_to_burnout,
                               compet_burnout_to_kpi,
                               perturbed_budget_constraints, total_budget, verbose=False)

    kpi1 = calc_kpi(x_new, q_new, expectations.a[selected, :],
                    expectations_to_burnout, compet_burnout_to_kpi)
    integral_kpi = np.dot(np.mean(kpi1, axis=0), compet_burnout_to_kpi.kpi_importance)
    #integral_kpi_list = np.array([integral_kpi])

    # еще 3 квартала
    z_quarterly_activities = np.sum(z, axis=0)
    for quart in range(2, 5):
        print("квартал", quart)
        z, x_new, q_new = optimize(x_new, q_new, expectations.a[selected, :],
                                   invest_to_compet, activities_expectations, expectations_to_burnout,
                                   compet_burnout_to_kpi,
                                   perturbed_budget_constraints, total_budget, verbose=False)
        z_quarterly_activities = np.c_[z_quarterly_activities, np.sum(z, axis=0)]
        kpi1 = calc_kpi(x_new, q_new, expectations.a[selected, :], expectations_to_burnout, compet_burnout_to_kpi)
        integral_kpi = np.dot(np.mean(kpi1, axis=0), compet_burnout_to_kpi.kpi_importance)
        #integral_kpi_list = np.c_[integral_kpi_list, integral_kpi]

    return integral_kpi, z_quarterly_activities.T


# Найти среднеквадратичный разброс при параметрах rel_shift, block_index с num_samples случайных реализаций
def calc_mean_std_constr(num_samples, rel_shift, block_index, file):
    fout_kpi = open(file, 'w')
    print("rel_shift =", rel_shift, ", block_index =", block_name[block_index] + ' (блок ' + str(block_index + 1) + ')', "\n", file=fout_kpi)
    kpi_sample = np.zeros(num_samples)
    Z = np.zeros((num_samples, num_activities))
    invest_by_compet_classes = np.zeros((num_samples, num_compet_classes))
    for i in range(num_samples):
        print("\n", "РЕАЛИЗАЦИЯ", i + 1, "\n")
        kpi_sample[i], Z[i, :], invest_by_compet_classes[i, :] = generate_random_kpi_constr(rel_shift, block_index)
        print(kpi_sample[i], file=fout_kpi)
        print(Z[i, :], file=fout_kpi)
        print(invest_by_compet_classes[i, :], file=fout_kpi)
    mu = np.mean(kpi_sample)
    sigma = np.std(kpi_sample)
    mu_Z = np.mean(Z, axis=0)
    sigma_Z = np.std(Z, axis=0)
    mu_classes = np.mean(invest_by_compet_classes, axis=0)
    sigma_classes = np.std(invest_by_compet_classes, axis=0)
    print("\n", "mu =", mu, ", sigma =", sigma, file=fout_kpi)
    print("mu_Z =", mu_Z, file=fout_kpi)
    print("sigma_Z =", sigma_Z, file=fout_kpi)
    print("mu_classes =", mu_classes, file=fout_kpi)
    print("sigma_classes =", sigma_classes, file=fout_kpi)

    mu_blocks = np.zeros(len(activities_blocks))
    for i, block in enumerate(activities_blocks):
        mu_blocks[i] = sum([mu_Z[j] for j in block])
    print("mu_blocks = ", mu_blocks, file=fout_kpi)

    fout_kpi.close()
    return mu, sigma, mu_Z, sigma_Z, mu_classes, sigma_classes


def calc_mean_std_constr_year(num_samples, rel_shift, block_index, file):
    fout_kpi = open(file, 'w')
    print("оптимизация за год -", "rel_shift =", rel_shift, ", block_index =", block_name[block_index] + ' (блок ' + str(block_index + 1) + ')', "\n", file=fout_kpi)
    kpi_sample = np.zeros(num_samples)
    Z = np.zeros((num_samples, 4, num_activities))
    for i in range(num_samples):
        print("\n", "РЕАЛИЗАЦИЯ", i + 1, "\n")
        kpi_sample[i], Z[i, :, :] = generate_random_kpi_constr_year(rel_shift, block_index)
        print(kpi_sample[i], file=fout_kpi)
        for quart in range(4):
            print(Z[i, quart, :], file=fout_kpi)

    mu = np.mean(kpi_sample)
    sigma = np.std(kpi_sample)
    mu_Z = np.mean(Z, axis=0)
    sigma_Z = np.std(Z, axis=0)
    print("\n", "mu =", mu, ", sigma =", sigma, file=fout_kpi)
    print("mu_Z =", mu_Z, file=fout_kpi)
    print("sigma_Z =", sigma_Z, file=fout_kpi)
    mu_Z_year = np.mean(np.sum(Z, axis=1), axis=0)
    sigma_Z_year = np.std(np.sum(Z, axis=1), axis=0)
    print("mu_Z_year =", mu_Z_year, file=fout_kpi)
    print("sigma_Z_year =", sigma_Z_year, file=fout_kpi)

    mu_blocks = np.zeros((4, len(activities_blocks)))
    for quart in range(4):
        for i, block in enumerate(activities_blocks):
            mu_blocks[quart, i] = sum([mu_Z[quart, j] for j in block])
    print("mu_blocks = ", mu_blocks, file=fout_kpi)

    mu_blocks_year = np.zeros(len(activities_blocks))
    for i, block in enumerate(activities_blocks):
        mu_blocks_year[i] = sum([mu_Z_year[j] for j in block])
    print("mu_blocks_year = ", mu_blocks_year, file=fout_kpi)

    fout_kpi.close()
    return mu, sigma, mu_Z, sigma_Z


def generate_random_kpi_data(var, perturb_compet=False, perturb_expect=False):
    perturbed_x = np.copy(compet_t0.x[selected, :])
    perturbed_q = np.copy(expectations.q[selected, :])

    if perturb_compet:
        for i in range(perturbed_x.shape[0]):
            for j in range(perturbed_x.shape[1]):
                perturbed_x[i, j] = perturb_coef(perturbed_x[i, j], var)

    if perturb_expect:
        for i in range(perturbed_q.shape[0]):
            for j in range(perturbed_q.shape[1]):
                perturbed_q[i, j] = perturb_coef(perturbed_q[i, j], var)

    z, x_new, q_new = optimize(perturbed_x, perturbed_q, expectations.a[selected, :],
                               invest_to_compet, activities_expectations, expectations_to_burnout,
                               compet_burnout_to_kpi,
                               budget_constraints, total_budget, verbose=False)

    kpi1 = calc_kpi(x_new, q_new, expectations.a[selected, :],
                    expectations_to_burnout, compet_burnout_to_kpi)
    integral_kpi = np.dot(np.mean(kpi1, axis=0), compet_burnout_to_kpi.kpi_importance)

    # распределение инвестиций по компетентностным классам
    invest_by_compet_classes = np.zeros(num_compet_classes)
    for j, i in enumerate(selected):
        u = compet_burnout_to_kpi.calc_u(2, compet_t0.x[i, :])  # берем категории для KPI с индексом 2
        #print(u)
        for k in range(num_compet_classes):
            invest_by_compet_classes[k] += u[k] * np.sum(z[j, :])

    return integral_kpi, np.sum(z, axis=0), invest_by_compet_classes


# Найти среднеквадратичный разброс при параметре var с num_samples случайных реализаций
def calc_mean_std_data(num_samples, var, perturb_compet, perturb_expect, file):
    fout_kpi = open(file, 'w')
    print("var =", var, ", perturb_compet =", perturb_compet, ", perturb_expect =", perturb_expect, "\n", file=fout_kpi)
    kpi_sample = np.zeros(num_samples)
    Z = np.zeros((num_samples, num_activities))
    invest_by_compet_classes = np.zeros((num_samples, num_compet_classes))
    for i in range(num_samples):
        print("\n", "РЕАЛИЗАЦИЯ", i + 1, "\n")
        kpi_sample[i], Z[i, :], invest_by_compet_classes[i, :] = generate_random_kpi_data(var, perturb_compet, perturb_expect)
        print(kpi_sample[i], file=fout_kpi)
        print(Z[i, :], file=fout_kpi)
        print(invest_by_compet_classes[i, :], file=fout_kpi)
    mu = np.mean(kpi_sample)
    sigma = np.std(kpi_sample)
    mu_Z = np.mean(Z, axis=0)
    sigma_Z = np.std(Z, axis=0)
    mu_classes = np.mean(invest_by_compet_classes, axis=0)
    sigma_classes = np.std(invest_by_compet_classes, axis=0)
    print("\n", "mu =", mu, ", sigma =", sigma, file=fout_kpi)
    print("mu_Z =", mu_Z, file=fout_kpi)
    print("sigma_Z =", sigma_Z, file=fout_kpi)
    print("mu_classes =", mu_classes, file=fout_kpi)
    print("sigma_classes =", sigma_classes, file=fout_kpi)

    mu_blocks = np.zeros(len(activities_blocks))
    for i, block in enumerate(activities_blocks):
        mu_blocks[i] = sum([mu_Z[j] for j in block])
    print("mu_blocks = ", mu_blocks, file=fout_kpi)

    fout_kpi.close()
    return mu, sigma, mu_Z, sigma_Z, mu_classes, sigma_classes


print("Реальные KPI при t = 0: ", np.mean(kpi_t0.y[selected, :], axis=0), " -> ",
    np.dot(np.mean(kpi_t0.y[selected, :], axis=0), compet_burnout_to_kpi.kpi_importance))

invest_to_compet_left_conf = InvestToCompet("invest_to_compet_left_conf_interval.csv")
invest_to_compet_right_conf = InvestToCompet("invest_to_compet_right_conf_interval.csv")


# Основной код
num_samples = 100
delta = 0.05
epsilon = 0.05
mu_kpi, sigma_kpi, mu_Z, sigma_Z, mu_classes, sigma_classes =\
    calc_mean_std(num_samples, delta, epsilon, file='kpi_realizations.txt')
print("mu = ", mu_kpi, " sigma = ", sigma_kpi)
print("mu_Z = ", mu_Z)
print("sigma_Z = ", sigma_Z)
print("mu_classes = ", mu_classes)
print("sigma_classes = ", sigma_classes)



# Сдвиг границ мероприятий
"""
num_samples = 100
block_index = 0
rel_shift = 0.1
mu_kpi, sigma_kpi, mu_Z, sigma_Z, mu_classes, sigma_classes =\
    calc_mean_std_constr(num_samples, rel_shift, block_index, file='kpi_realizations.txt')
print("mu = ", mu_kpi, " sigma = ", sigma_kpi)
print("mu_Z = ", mu_Z)
print("sigma_Z = ", sigma_Z)
print("mu_classes = ", mu_classes)
print("sigma_classes = ", sigma_classes)

mu_blocks = np.zeros(len(activities_blocks))
for i, block in enumerate(activities_blocks):
    mu_blocks[i] = sum([mu_Z[j] for j in block])
print("mu_blocks = ", mu_blocks)
"""


# Сдвиг границ мероприятий (оптимизация за год)
"""
num_samples = 100
block_index = 0
rel_shift = 0.1
#mu_kpi, sigma_kpi, mu_Z, sigma_Z, mu_classes, sigma_classes =\
calc_mean_std_constr_year(num_samples, rel_shift, block_index, file='kpi_realizations.txt')
"""

"""
num_samples = 100
block_index = 0
rel_shift = 0.1
calc_mean_std_constr_year(num_samples, rel_shift, block_index, file='Случайные реализации/Сдвиг границ_за год/10_1_год.txt')

num_samples = 100
block_index = 1
rel_shift = 0.1
calc_mean_std_constr_year(num_samples, rel_shift, block_index, file='Случайные реализации/Сдвиг границ_за год/10_2_год.txt')

num_samples = 100
block_index = 2
rel_shift = 0.1
calc_mean_std_constr_year(num_samples, rel_shift, block_index, file='Случайные реализации/Сдвиг границ_за год/10_3_год.txt')

num_samples = 100
block_index = 3
rel_shift = 0.1
calc_mean_std_constr_year(num_samples, rel_shift, block_index, file='Случайные реализации/Сдвиг границ_за год/10_4_год.txt')

num_samples = 100
block_index = 4
rel_shift = 0.1
calc_mean_std_constr_year(num_samples, rel_shift, block_index, file='Случайные реализации/Сдвиг границ_за год/10_5_год.txt')
"""

"""
print("mu = ", mu_kpi, " sigma = ", sigma_kpi)
print("mu_Z = ", mu_Z)
print("sigma_Z = ", sigma_Z)
print("mu_classes = ", mu_classes)
print("sigma_classes = ", sigma_classes)

mu_blocks = np.zeros(len(activities_blocks))
for i, block in enumerate(activities_blocks):
    mu_blocks[i] = sum([mu_Z[j] for j in block])
print("mu_blocks = ", mu_blocks)
"""


# Варьируем данные
"""
num_samples = 100
var = 0.1
mu_kpi, sigma_kpi, mu_Z, sigma_Z, mu_classes, sigma_classes =\
    calc_mean_std_data(num_samples, var, perturb_compet=True, perturb_expect=False, file='kpi_realizations.txt')
print("mu = ", mu_kpi, " sigma = ", sigma_kpi)
print("mu_Z = ", mu_Z)
print("sigma_Z = ", sigma_Z)
print("mu_classes = ", mu_classes)
print("sigma_classes = ", sigma_classes)

mu_blocks = np.zeros(len(activities_blocks))
for i, block in enumerate(activities_blocks):
    mu_blocks[i] = sum([mu_Z[j] for j in block])
print("mu_blocks = ", mu_blocks)
"""
