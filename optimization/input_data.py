import numpy as np

# число компетенций
num_compet = 38
# число направлений мероприятий
num_activities = 29
# число показателей выгорания
num_burnout_indicators = 3

# число диапазонов интегрального показателя ожиданий
num_expectation_classes = 5

# единицы расчета стоимостей (1000 - значит в тысячах рублей)
RUB_COEFF = 1000


class InvestToCompet:
    # x_ij - j-я компетенция i-го сотрудника
    # s_im - вложения по m-му направлению по отношению к i-му сотруднику
    # x_ij(t+1) = beta_j * x_ij(t) + sum_m[ alpha_jm * s_im(t) ]
    def __init__(self, filename):
        # коэффициенты в единицах рублей
        # palpha - на сколько увеличится компетенция при вложении одного рубля
        self.palpha = np.zeros((num_compet, num_activities))
        # pbeta - коэффициент при компетенции в предыдущий момент времени
        self.pbeta = np.zeros(num_compet)
        # alpha - на сколько увеличится компетенция при вложении 1000 рублей
        self.alpha = self.palpha * RUB_COEFF
        # beta - коэффициент при компетенции в предыдущий момент времени
        self.beta = self.pbeta
        self.read(filename)
        
    def read(self, filename):
        pass


class ActivitiesExpectations:
    # q_im - отклонение ожиданий i-го сотрудника от реализации m-го направления мероприятий
    # mu_m - минимально ожидаемые инвестиции
    # nu_m - максимально ожидаемые инвестиции
    # q_im(t+1) = max(min(q_im(t) + 2 * (s_im(t) - mu_m) / (nu_m - mu_m), 1), -1)
    def __init__(self, filename):
        self.mu = np.zeros(num_activities)
        self.nu = np.zeros(num_activities)
        self.read(filename)
        
    def read(self, filename):
        # при чтении разделить mu и nu на RUB_COEFF и умножить на 3 (число месяцев в квартале)
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
    # b_il(t) = phi_l(gamma_l + sum_m[ delta_lm * (a_im * q_im(t)) ]
    # b_il - l-й показатель i-го сотрудника
    # gamma_l + sum_m[ delta_lm * (a_im * q_im(t)) - интегральный показатель ожиданий
    # phi_l - кусочно-линейная функция, аппроксимирующая матрицу соответствий между
    #   диапазонами интегрального показателя ожиданий и диапазонами показателя выгорания
    # t_p - границы диапазонов интегрального показателя ожиданий
    #   t_p (p = 0..num_expectation_classes-2) - границы между p-м и (p+1)-м диапазонами
    # как-то обозначить равномерные диапазоны значений показателя выгорания
    def __init__(self, filename):
        self.gamma = np.zeros(num_burnout_indicators)
        self.delta = np.zeros((num_burnout_indicators, num_activities))
        # инициализировать self.t
        self.read(filename)
        
    def read(self, filename):
        pass
    # рассчитать кусочно-линейную зависимость phi_l, заданную координатами узловых точек
        
    # рассчитать l-й показатель выгорания, зная:
    #   a_m - вектор важности для сотрудника каждого m-го направления мероприятий
    #   q_m - вектор отклонений ожиданий сотрудника от реализации m-го направления мероприятий
    def calc_burnout(self, l, a, q):
        pass
        
    # внести в этот класс диапазоны интегрального показателя
    # также добавить функцию расчета коэффициентов линеаризованной модели
    #   (в рамках текущего диапазона интегрального показателя), если заданы векторы a, q
