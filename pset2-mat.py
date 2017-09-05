import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import norm, lognorm, chisquare

# Exercise 2.1
filename = 'body_men.mat'
X = loadmat(filename)['body_men'][:,1]
mu = np.mean(X)
si = np.std(X)

print("Mean: %f" % mu)
print("Std: %f" % si)

logX = np.log(X)
logmu = np.mean(logX)
logsi = np.std(logX)

print("Log Mean: %f" % logmu)
print("Log Std: %f" % logsi)

# Let's do chi-squared as well! See how well each one fits
# Try just 4 buckets for now, based on normal dist
binsq = [0, 0.25, 0.5, 0.75, 1]
# Can also try 10, or more buckets. Fewer buckets => "looks better"!
#binsq = np.linspace(0, 1, 11)
bins = norm.ppf(binsq, mu, si)

hist, bin_edges = np.histogram(X, bins)

print("Histogram:")
print(hist)

c2, p = chisquare(hist)
print("Chi-Square: %f (p-value %f)" % (c2, p))

# Plot normal and log-normal approximations to X, along with histogram for X
x = np.linspace(np.amin(X)-5, np.amax(X)+5, 100)
plt.plot(x, norm.pdf(x, mu, si))
plt.plot(x, lognorm.pdf(x, 1, mu, si))

plt.hist(X, bins='auto', normed=True)
plt.show()
# WOW this histogram is way messier than the other one! Fewer buckets -> better

