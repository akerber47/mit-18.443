import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import norm, lognorm, chisquare, ttest_ind, f, bartlett, levene, probplot

# Exercise 2.1
filename = 'normtemp.mat'
data = loadmat(filename)['normtemp']
X = data[:64,2]
Y = np.random.choice(data[65:,2], 50)

mu_x = np.mean(X)
si_x = np.std(X)
print("Mean (X): %f" % mu_x)
print("Std (X): %f" % si_x)

mu_y = np.mean(Y)
si_y = np.std(Y)
print("Mean (Y): %f" % mu_y)
print("Std (Y): %f" % si_y)

print("4 bins:")
binsq = [0, 0.25, 0.5, 0.75, 1]
bins_x = norm.ppf(binsq, mu_x, si_x)
bins_y = norm.ppf(binsq, mu_y, si_y)

hist, bin_edges = np.histogram(X, bins_x)
c2, p = chisquare(hist)
print("Chi-Square (X): %f (p-value %f)" % (c2, p))

hist, bin_edges = np.histogram(Y, bins_y)
c2, p = chisquare(hist)
print("Chi-Square (Y): %f (p-value %f)" % (c2, p))

print("8 bins:")
binsq = np.linspace(0, 1, 9)
bins_x = norm.ppf(binsq, mu_x, si_x)
bins_y = norm.ppf(binsq, mu_y, si_y)

hist, bin_edges = np.histogram(X, bins_x)
c2, p = chisquare(hist)
print("Chi-Square (X): %f (p-value %f)" % (c2, p))

hist, bin_edges = np.histogram(Y, bins_y)
c2, p = chisquare(hist)
print("Chi-Square (Y): %f (p-value %f)" % (c2, p))

t, tp = ttest_ind(X, Y)
print("t test with equal variance (X,Y): %f (p-value %f)" % (t, tp))
t, tp = ttest_ind(X, Y, equal_var=False)
print("t test with unequal variance (X,Y): %f (p-value %f)" % (t, tp))

F = si_x / si_y
Fp = f.cdf(F, 65, 50)
print("F test (X,Y): %f (p-value %f)" % (F, Fp))
# Just for fun, try a couple other tests too, which are more robust
B, Bp = bartlett(X, Y)
print("Bartlett's test for equal variances (X,Y): %f (%f)" % (B, Bp))
L, Lp = levene(X, Y)
print("Levene's test for equal variances (X,Y): %f (%f)" % (L, Lp))

# Let's draw some QQ plots too because WHY NOT
plt.subplot(221)
probplot(X, plot=plt)
plt.subplot(222)
probplot(X, plot=plt)
plt.subplot(223)
probplot(Y, plot=plt)
plt.subplot(224)
probplot(Y, plot=plt)
plt.show()
