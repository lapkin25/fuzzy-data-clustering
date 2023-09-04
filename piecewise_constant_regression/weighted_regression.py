# Минимизация взвешенной суммы квадратов отклонений

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# Возвращает два объекта:
#   массив коэффициентов при признаках
#   и число - свободный член
def calc_weighted_regression(x, y, u):
    # обычная регрессия - проверка
    #print(x)
    #print(y)
    #reg = LinearRegression().fit(x, y)
    #w = reg.coef_
    #w0 = reg.intercept_
    #print("w =", w)
    #print("w0 =", w0)
    #x1 = x.copy()
    x1 = x[u != 0]
    #y1 = y.copy()
    y1 = y[u != 0]
    u1 = u[u != 0]
    x1 = sm.add_constant(x1)
    #reg1 = sm.OLS(y1, x1).fit()
    #print(reg1.summary())

    reg_weighted = sm.WLS(y1, x1, weights=u1).fit()
    #print(u1)
    #reg_weighted = sm.WLS(y1, x1).fit()
    #print(reg_weighted.summary())
    #print(reg_weighted.params)
    params = reg_weighted.params
    return params[1:], params[0]
