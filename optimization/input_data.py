import numpy as np
import csv

# число наблюдений
data_size = 219

# число компетенций
num_compet = 38
# число направлений мероприятий
num_activities = 29
# число показателей выгорания
num_burnout_indicators = 3
# число показателей KPI
num_kpi_indicators = 4

# число диапазонов интегрального показателя ожиданий
num_expectation_classes = 5
# число диапазонов интегрального показателя компетентности
num_compet_classes = 6


class InvestToCompet:
    # x_ij - j-я компетенция i-го сотрудника
    # s_im - вложения по m-му направлению по отношению к i-му сотруднику
    # x_ij(t+1) = beta_j * x_ij(t) + sum_m[ alpha_jm * s_im(t) ]
    def __init__(self, file_name):
        # коэффициенты в единицах рублей
        # alpha - на сколько увеличится компетенция при вложении одного рубля
        self.alpha = np.zeros((num_compet, num_activities))
        # beta - коэффициент при компетенции в предыдущий момент времени
        self.beta = np.zeros(num_compet)
        self.read(file_name)

    def read(self, file_name):
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == num_compet)
        for j, row in enumerate(data_str):
            row_coeff = row[1:]
            assert(len(row_coeff) == num_activities + 1)
            self.beta[j] = float(row_coeff[0])
            self.alpha[j, :] = np.array(list(map(float, row_coeff[1:])))


class CompetData:
    # x_ij - j-я компетенция i-го сотрудника
    def __init__(self, file_name):
        self.x = np.zeros((data_size, num_compet))
        self.read(file_name)

    def read(self, file_name):
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == data_size)
        for i, row in enumerate(data_str):
            assert(len(row) == num_compet)
            self.x[i, :] = np.array(list(map(float, row)))


class BurnoutData:
    # b_il - l-й показатель выгорания i-го сотрудника
    def __init__(self, file_name):
        self.b = np.zeros((data_size, num_burnout_indicators))
        self.read(file_name)

    def read(self, file_name):
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == data_size)
        for i, row in enumerate(data_str):
            assert(len(row) == num_burnout_indicators)
            self.b[i, :] = np.array(list(map(float, row)))


class KPIData:
    # y_im - m-й KPI i-го сотрудника
    def __init__(self, file_name):
        self.y = np.zeros((data_size, num_kpi_indicators))
        self.read(file_name)

    def read(self, file_name):
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == data_size)
        for i, row in enumerate(data_str):
            assert(len(row) == num_kpi_indicators)
            self.y[i, :] = np.array(list(map(float, row)))


class ExpectationsData:
    # q_im - отклонение ожиданий i-го сотрудника от реализации m-го направления мероприятий
    # a_im - важность для i-го сотрудника m-го направления мероприятий
    def __init__(self, file_name):
        self.q = np.zeros((data_size, num_activities))
        self.a = np.zeros((data_size, num_activities))
        self.read(file_name)

    def read(self, file_name):
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == data_size)
        for i, row in enumerate(data_str):
            assert(len(row) == num_activities)
            for m in range(num_activities):
                val = float(row[m])
                if val > 0:
                    self.a[i, m] = val
                    self.q[i, m] = 1.0
                elif val < 0:
                    self.a[i, m] = -val
                    self.q[i, m] = -1.0
                else:  # val == 0
                    self.a[i, m] = 0.0
                    self.q[i, m] = 0.0


class ActivitiesExpectations:
    # q_im - отклонение ожиданий i-го сотрудника от реализации m-го направления мероприятий
    # mu_m - минимально ожидаемые инвестиции за квартал в человека
    # nu_m - максимально ожидаемые инвестиции за квартал в человека
    # q_im(t+1) = max(min(q_im(t) + 2 * (s_im(t) - mu_m) / (nu_m - mu_m), 1), -1)
    def __init__(self, file_name):
        self.mu = np.zeros(num_activities)
        self.nu = np.zeros(num_activities)
        self.read(file_name)
        
    def read(self, file_name):
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == num_activities)
        for k, row in enumerate(data_str):
            assert(len(row) == 3)
            # умножаем на 3, так как в файле инвестиции за месяц, а надо - за квартал
            self.mu[k] = float(row[1]) * 3
            self.nu[k] = float(row[2]) * 3

    # рассчитать отклонения ожиданий в момент времени (t+1),
    #   зная: k - номер направления мероприятий, s - объем инвестиций,
    #         qt - отклонения ожиданий в момент времени t
    def calc_expectations(self, k, s, qt):
        qnext = qt + 2 * (s - self.mu[k]) / (self.nu[k] - self.mu[k])
        if qnext < -1:
            qnext = -1
        if qnext > 1:
            qnext = 1
        return qnext


class ExpectationsToBurnout:
    # b_il(t) = psi_l(sum_m[ w_lm * (a_im * q_im(t)) ])
    # b_il - l-й показатель i-го сотрудника
    # sum_m[ w_lm * (a_im * q_im(t)) ] - интегральный показатель ожиданий
    # psi_l - кусочно-линейная функция, аппроксимирующая матрицу соответствий между
    #   диапазонами интегрального показателя ожиданий и диапазонами показателя выгорания
    # t_p - границы диапазонов интегрального показателя ожиданий
    #   t_p (p = 1..num_expectation_classes-1) - границы между p-м и (p+1)-м диапазонами
    # r_p - центры диапазонов изменения интегрального показателя ожиданий
    # e_p - ожидаемые значения показателя выгорания в центрах r_p
    # равномерные диапазоны значений показателя выгорания
    def __init__(self, expectations_data):
        self.w = np.zeros((num_burnout_indicators, num_activities))
        self.t = np.zeros((num_burnout_indicators, num_expectation_classes + 1))
        self.r = np.zeros((num_burnout_indicators, num_expectation_classes))
        self.e = np.zeros((num_burnout_indicators, num_expectation_classes))
        self.corr_matrix = np.zeros((num_burnout_indicators, num_expectation_classes, num_expectation_classes))
        self.read(expectations_data)
        
    def read(self, expectations_data):
        for l in range(num_burnout_indicators):
            file_name = "expect_to_burnout_" + str(l + 1) + "_matrix.csv"
            with open(file_name) as fp:
                reader = csv.reader(fp, delimiter=";")
                next(reader, None)  # пропустить заголовки
                data_str = [row for row in reader]
            assert(len(data_str) == num_expectation_classes)
            for i, row in enumerate(data_str):
                row = row[1:]
                assert(len(row) == num_expectation_classes)
                self.corr_matrix[l, i, :] = np.array(list(map(float, row)))

            file_name = "expect_to_burnout_" + str(l + 1) + "_weights.csv"
            with open(file_name) as fp:
                reader = csv.reader(fp, delimiter=";")
                data_str = [row for row in reader]
            assert(len(data_str) == 1)
            row = data_str[0]
            assert(len(row) == num_activities)
            self.w[l, :] = np.array(list(map(float, row)))

            file_name = "expect_to_burnout_" + str(l + 1) + "_intervals.csv"
            with open(file_name) as fp:
                reader = csv.reader(fp, delimiter=";")
                data_str = [row for row in reader]
            assert(len(data_str) == 1)
            row = data_str[0]
            assert(len(row) == num_expectation_classes - 1)
            self.t[l, 1:num_expectation_classes] = np.array(list(map(float, row)))

            integral_expectations = np.dot(expectations_data.q * expectations_data.a, self.w[l, :])
            self.t[l, 0] = np.min(integral_expectations)
            self.t[l, num_expectation_classes] = np.max(integral_expectations)

        y_min = 0
        y_max = 100
        avg_y = np.array([(i + 0.5) * (y_max - y_min) / num_expectation_classes
                          for i in range(num_expectation_classes)])
        for l in range(num_burnout_indicators):
            self.r[l, :] = np.array([(self.t[l, i] + self.t[l, i + 1]) / 2 for i in range(num_expectation_classes)])
            self.e[l, :] = np.dot(self.corr_matrix[l, :, :], avg_y)

    # рассчитать l-й показатель выгорания, зная:
    #   a_m - вектор важности для сотрудника каждого m-го направления мероприятий
    #   q_m - вектор отклонений ожиданий сотрудника от реализации m-го направления мероприятий
    def calc_burnout(self, l, a, q):
        integral_expectations = np.dot(q * a, self.w[l, :])
        if integral_expectations < self.r[l, 0]:
            return self.e[l, 0]
        elif integral_expectations > self.r[l, num_expectation_classes - 1]:
            return self.e[l, num_expectation_classes - 1]
        else:
            # найдем ближайшую справа точку излома
            p = 0
            while integral_expectations > self.r[l, p]:
                p += 1
            # p - номер ближайшей справа точки излома
            # усредняем e[l, p - 1] и e[l, p]
            lam = (integral_expectations - self.r[l, p - 1]) / (self.r[l, p] - self.r[l, p - 1])
            return self.e[l, p - 1] * (1 - lam) + self.e[l, p] * lam

        
    # TODO: добавить функцию расчета коэффициентов линеаризованной модели
    #   (в рамках текущего диапазона интегрального показателя), если заданы векторы a, q


class CompetBurnoutToKPI:
    # x_ij - j-я компетенция i-го сотрудника
    # b_il - l-й показатель выгорания i-го сотрудника
    # y_im - m-й KPI i-го сотрудника
    # y_im(t) = phi_m(sum_j[ w_mj * x_ij(t) ]) + W_m0 + W_m1 * b_i1(t) + W_m2 * b_i2(t) + W_m3 * b_i3(t)
    # sum_j[ w_mj * x_ij(t) ] - интегральный показатель компетентности
    # phi_m - кусочно-линейная функция, выражающая зависимость KPI от интегрального показателя компетентности
    # t_p - границы диапазонов интегрального показателя компетентности
    #   t_p (p = 1..num_compet_classes-1) - границы между p-м и (p+1)-м диапазонами
    # c_p - центры диапазонов изменения KPI
    # r_p - центры диапазонов изменения интегрального показателя компетентности

    def __init__(self, compet_data):
        self.w = np.zeros((num_kpi_indicators, num_compet))
        self.t = np.zeros((num_kpi_indicators, num_compet_classes + 1))
        self.c = np.zeros((num_kpi_indicators, num_compet_classes))
        self.r = np.zeros((num_kpi_indicators, num_compet_classes))
        self.kpi_importance = np.zeros(num_kpi_indicators)
        self.burnout_intercept = np.zeros(num_kpi_indicators)
        self.burnout_coef = np.zeros((num_kpi_indicators, num_burnout_indicators))
        self.read(compet_data)

    def read(self, compet_data):
        file_name = "compet_to_kpi_weights.csv"
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == num_kpi_indicators)
        for m, row in enumerate(data_str):
            row = row[1:]
            assert(len(row) == num_compet)
            self.w[m, :] = np.array(list(map(float, row)))

        file_name = "compet_to_kpi_intervals.csv"
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == num_kpi_indicators)
        for m, row in enumerate(data_str):
            row = row[1:]
            assert(len(row) == num_compet_classes - 1)
            self.t[m, 1:num_compet_classes] = np.array(list(map(float, row)))

        for m in range(num_kpi_indicators):
            integral_compet = np.dot(compet_data.x, self.w[m, :])
            self.t[m, 0] = np.min(integral_compet)
            self.t[m, num_compet_classes] = np.max(integral_compet)

        file_name = "compet_to_kpi_average.csv"
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == num_kpi_indicators)
        for m, row in enumerate(data_str):
            row = row[1:]
            assert(len(row) == num_compet_classes)
            self.c[m, :] = np.array(list(map(float, row)))

        for m in range(num_kpi_indicators):
            self.r[m, :] = np.array([(self.t[m, i] + self.t[m, i + 1]) / 2 for i in range(num_compet_classes)])

        file_name = "burnout_to_kpi.csv"
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == num_kpi_indicators)
        for m, row in enumerate(data_str):
            row = row[1:]
            assert(len(row) == num_burnout_indicators + 1)
            self.burnout_intercept[m] = float(row[0])
            self.burnout_coef[m, :] = np.array(list(map(float, row[1:])))

        file_name = "kpi_importance.csv"
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            data_str = [row for row in reader]
        assert(len(data_str) == 1)
        row = data_str[0]
        assert(len(row) == num_kpi_indicators)
        self.kpi_importance = np.array(list(map(float, row)))

    # вычислить зависимость m-го KPI от компетенций для отдельного сотрудника
    def calc_phi(self, m, x):
        integral_compet = np.dot(x, self.w[m, :])
        if integral_compet < self.r[m, 0]:
            return self.c[m, 0]
        elif integral_compet > self.r[m, num_compet_classes - 1]:
            return self.c[m, num_compet_classes - 1]
        else:
            # найдем ближайшую справа точку излома
            p = 0
            while integral_compet > self.r[m, p]:
                p += 1
            # p - номер ближайшей справа точки излома
            # усредняем c[m, p - 1] и c[m, p]
            lam = (integral_compet - self.r[m, p - 1]) / (self.r[m, p] - self.r[m, p - 1])
            return self.c[m, p - 1] * (1 - lam) + self.c[m, p] * lam

    # вычислить m-й KPI, зная компетенции и выгорание отдельного сотрудника
    def calc_kpi(self, m, x, b):
        return self.calc_phi(m, x) + self.burnout_intercept[m] + np.dot(self.burnout_coef[m, :], b)

    # вычислить меры принадлежности компетентностным категориям m-го KPI, зная вектор компетенций x
    def calc_u(self, m, x):
        integral_compet = np.dot(x, self.w[m, :])
        return np.array([self.calc_u_k_given_a(k, integral_compet, self.r[m, :]) for k in range(num_compet_classes)])

    def calc_u_k_given_a(self, k, x, a):
        m = len(a)  # количество диапазонов
        if k == 0:
            if x <= a[0]:
                u_val = 1
            elif x > a[0] and x <= a[1]:
                u_val = (a[1] - x) / (a[1] - a[0])
            else:
                u_val = 0
        elif k == m - 1:
            if x >= a[m - 1]:
                u_val = 1
            elif x >= a[m - 2] and x < a[m - 1]:
                u_val = (x - a[m - 2]) / (a[m - 1] - a[m - 2])
            else:
                u_val = 0
        else:  # 0 < k < m - 1
            if x >= a[k - 1] and x <= a[k]:
                u_val = (x - a[k - 1]) / (a[k] - a[k - 1])
            elif x >= a[k] and x <= a[k + 1]:
                u_val = (a[k + 1] - x) / (a[k + 1] - a[k])
            else:
                u_val = 0
        return u_val


class BudgetConstraints:
    def __init__(self, file_name):
        self.budget_activities_percent = np.zeros(num_activities)
        self.read(file_name)

    def read(self, file_name):
        with open(file_name) as fp:
            reader = csv.reader(fp, delimiter=";")
            next(reader, None)  # пропустить заголовки
            data_str = [row for row in reader]
        assert(len(data_str) == 1)
        row = data_str[0]
        assert(len(row) == num_activities)
        self.budget_activities_percent = np.array(list(map(float, row)))
