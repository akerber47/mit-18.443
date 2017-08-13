import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm

#Exercise 1.3
X = np.random.normal(5, 2, 100)
print(X)

mean = np.mean(X)
print("Mean: %f" % mean)

std = np.std(X)
print("Std: %f" % std)
# This is equivalent to std(X,1) in matlab

std1 = np.std(X, ddof=1)
print("Std, ddof=1: %f" % std1)
# This is equivalent to std(X) in matlab

x = np.linspace(0,10, 100)
# empirical cdf
ecdf = ECDF(X)
plt.plot(x, ecdf(x))
# normal cdf
plt.plot(x, norm.cdf(x, 5, 2))
plt.show()
