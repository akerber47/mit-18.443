import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

# Exercise 2.1
filename = 'heights.txt'
X = np.genfromtxt(filename, skip_header=1)
mu = np.mean(X)
si = np.std(X)

print("Mean: %f" % mu)
print("Std: %f" % si)

logX = np.log(X)
logmu = np.mean(logX)
logsi = np.std(logX)

print("Log Mean: %f" % logmu)
print("Log Std: %f" % logsi)

# Plot normal and log-normal approximations to X, along with histogram for X
x = np.linspace(60, 80, 100)
plt.plot(x, norm.pdf(x, mu, si))
plt.plot(x, lognorm.pdf(x, 1, mu, si))

plt.hist(X, 50, normed=True)
plt.show()

# Will skip chi-squared for now, because I'm pretty sure that comes much later
# in the class. Also because I'm confused... :(
