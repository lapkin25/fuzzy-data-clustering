# Минимизация взвешенной суммы квадратов отклонений

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import math


# Принимает на вход:
#   двумерный массив x (x[i, k] - k-я компонента i-й точки)
#   одномерный массив y (y[i] - координата i-й точки)
#   веса (u[i] - вес i-й точки)
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

    reg_weighted = sm.WLS(y1, x1, weights=u1)
    reg_weighted_fitted = reg_weighted.fit()
    #print(u1)
    #reg_weighted = sm.WLS(y1, x1).fit()
    print(reg_weighted_fitted.summary())

    from statsmodels.stats.outliers_influence import summary_table
    st, data, ss2 = summary_table(reg_weighted_fitted, alpha=0.05)
    print(st, '\n', type(st))
    print(data[:, 10])
    print('Weighted Studentized residuals:')#, data[:, 10], np.sqrt(u1))
    for i in range(len(u1)):
        print(data[i, 10] * math.sqrt(u1[i]))
        if (data[i, 10] * math.sqrt(u1[i]) > 3):
            print('!!!')

    print(data[:, 11])
    print('Mean D =', np.mean(data[:, 11]))
    #print('Outliers: ', np.sum(data[:, 11] > 3 * np.mean(data[:, 11])), 'of', len(u1))
    print('Outliers: ', np.sum(data[:, 11] > 3 * np.mean(data[:, 11])), 'of', len(u1))
    print('Outliers 2: ', np.sum(data[:, 11] > 4 / np.sum(u1)), 'of', len(u1))
    #print(f'D_crit = 4/n = {4 / reg_weighted_fitted.nobs}')
    print('sum u =', np.sum(u1))
    print(f'D_crit = 4/n = {4 / np.sum(u1)}')
    print('DFFIT threshold =', 2 * math.sqrt(5) / (np.sum(u1) - 5))
   # for i in range(len(u1)):
   #     print(i + 1, u1[i])

    #print(reg_weighted.params)
    params = reg_weighted_fitted.params
    #print(reg_weighted.predict(params))
    print("взвешенная сумма квадратов отклонений =", np.sum(np.dot(u1, (reg_weighted.predict(params) - y1) ** 2)))
    print("корень из среднего квадрата =", math.sqrt(np.sum(np.dot(u1, (reg_weighted.predict(params) - y1) ** 2)) / np.sum(u1)))


    import pylab 
    import scipy.stats as stats 
    stats.probplot(y1-reg_weighted.predict(params), dist="norm", plot=pylab) 
    pylab.title('QQ Plot: Test Gaussian Residuals') 
    pylab.show()

    res = stats.shapiro(y1-reg_weighted.predict(params))
    print('Shapiro-Wilk test: ', res)
    


    return params[1:], params[0]


def plot_weighted_regression(x, y, u, w, w0):
    wsum_x = w0 + np.dot(x, np.transpose(w))
    plt.scatter(wsum_x, y, c=u, cmap='Reds')
    plt.plot([np.min(wsum_x), np.max(wsum_x)], [np.min(wsum_x), np.max(wsum_x)])
    plt.xlabel("Integral burnout indicator")
    plt.ylabel("KPI")
    plt.savefig('fig2.png', dpi=300)
    plt.show()
