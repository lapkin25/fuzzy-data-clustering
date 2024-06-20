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

# число диапазонов интегрального показателя ожиданий
num_expectation_classes = 5


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
    # mu_m - минимально ожидаемые инвестиции
    # nu_m - максимально ожидаемые инвестиции
    # q_im(t+1) = max(min(q_im(t) + 2 * (s_im(t) - mu_m) / (nu_m - mu_m), 1), -1)
    def __init__(self, file_name):
        self.mu = np.zeros(num_activities)
        self.nu = np.zeros(num_activities)
        self.read(file_name)
        
    def read(self, file_name):
        # при чтении умножить числа на 3 (число месяцев в квартале)
        pass

    # рассчитать отклонения ожиданий в момент времени (t+1),
    #   зная: m - номер направления мероприятий, s - объем инвестиций,
    #         qt - отклонения ожиданий в момент времени t
    def calc_expectations(self, m, s, qt):
        qnext = qt + 2 * (s - self.mu[m]) / (self.nu[m] - self.mu[m])
        if qnext < -1:
            qnext = -1
        if qnext > 1:
            qnext = 1
        return qnext


class ExpectationsToBurnout:
    # b_il(t) = phi_l(sum_m[ w_lm * (a_im * q_im(t)) ])
    # b_il - l-й показатель i-го сотрудника
    # sum_m[ w_lm * (a_im * q_im(t)) ] - интегральный показатель ожиданий
    # phi_l - кусочно-линейная функция, аппроксимирующая матрицу соответствий между
    #   диапазонами интегрального показателя ожиданий и диапазонами показателя выгорания
    # t_p - границы диапазонов интегрального показателя ожиданий
    #   t_p (p = 0..num_expectation_classes-2) - границы между p-м и (p+1)-м диапазонами
    # равномерные диапазоны значений показателя выгорания
    def __init__(self, expectations_data):
        self.w = np.zeros((num_burnout_indicators, num_activities))
        self.t = np.zeros((num_burnout_indicators, num_expectation_classes + 1))
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


    # рассчитать кусочно-линейную зависимость phi_l, заданную координатами узловых точек
        
    # рассчитать l-й показатель выгорания, зная:
    #   a_m - вектор важности для сотрудника каждого m-го направления мероприятий
    #   q_m - вектор отклонений ожиданий сотрудника от реализации m-го направления мероприятий
    def calc_burnout(self, l, a, q):
        pass
        
    # внести в этот класс диапазоны интегрального показателя
    # также добавить функцию расчета коэффициентов линеаризованной модели
    #   (в рамках текущего диапазона интегрального показателя), если заданы векторы a, q
