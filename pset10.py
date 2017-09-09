import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import norm, lognorm, chisquare, ttest_ind, f, bartlett, levene, probplot
# Let's cheat (because this is what we'd do in real life).
# Can't get the p-value/variance estimates this way, but I'm not at all sure
# those numbers mean anything anyway...
from sklearn import linear_model

# Exercise 2.1
filename = 'oecd.mat'
data = loadmat(filename)['eocd'] # typo in data file
Y = np.log(data[:,0])
X = data[:,1:3]

print("Basic linear regression")
model = linear_model.LinearRegression().fit(X, Y)
print("Coefficients: beta1 = %f, beta2 = %f" % (model.coef_[0],model.coef_[1]))
print("Intercept: beta0 = %f" % model.intercept_)

# Score it. No built-ins for frequentist stuff (confidence intervals etc)?
# Maybe implement later? Or maybe no one uses them anymore...
# Maybe in scipy under ANOVA?
print("Score: R^2 = %f" % model.score(X, Y))

# Now let's try some fun stuff!

print("\nRidge regression")
model = linear_model.Ridge(alpha=0.5).fit(X,Y)
print("Coefficients: beta1 = %f, beta2 = %f" % (model.coef_[0],model.coef_[1]))
print("Intercept: beta0 = %f" % model.intercept_)
print("Score: R^2 = %f" % model.score(X, Y))

print("\nRidge regression with built-in cross-validation")
model = linear_model.RidgeCV(alphas=(0.1,0.5,1.0,5,10,50,100,500),
        store_cv_values=True).fit(X,Y)

# Not really sure what this is doing ... gotta study more!
print("Estimated alpha: %f" % model.alpha_)
print("CV values:")
print(model.cv_values_)
print("Coefficients: beta1 = %f, beta2 = %f" % (model.coef_[0],model.coef_[1]))
print("Intercept: beta0 = %f" % model.intercept_)
print("Score: R^2 = %f" % model.score(X, Y))
