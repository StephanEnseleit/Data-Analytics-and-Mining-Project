import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.linalg import inv
from numpy.linalg import det
from numpy import dot
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import skew
from scipy.stats import pearsonr

sns.set_style("darkgrid")

def ss_total(y):
    return np.sum((y - np.mean(y))**2)

# The regression sum of squares, also called the explained sum of squares:
def ss_reg(pred, y):
    return np.sum((pred - np.mean(y))**2)

# The sum of squares of residuals, also called the residual sum of squares:
def ss_res(pred, y):
    return np.sum((y - pred)**2) 

df = pd.read_csv('../../communities_processed.csv')
corr_matrix = df.corr()
corr_matrix.to_csv('correlation_matrix_crimes.csv')

independent_X = df['TotalPctDiv'].to_numpy().reshape(-1,1) 
dependent_Y = df['PctPopUnderPov'].to_numpy()

ax = sns.regplot(x=independent_X, y=dependent_Y, fit_reg=False, scatter_kws={'alpha':0.4})
ax.set(xlabel='percentage of people being divorced', ylabel='percentage of people in poverty', title='')

#plt.show()

# Linear Regression
## add ones vector
ones_vector = np.ones(len(independent_X))
independent_X = np.c_[ones_vector, independent_X]
## perform regression
beta = np.linalg.lstsq(independent_X, dependent_Y, rcond=None)[0]
print('Estimated coefficients:', beta)
predictions_with_intercept = dot(independent_X, beta) 
## visualize fitting
plt.plot(independent_X[:,1], predictions_with_intercept)


#plt.show()

# Model Fitting
## calculate rsq
sstotal = ss_total(dependent_Y)
ssreg = ss_reg(predictions_with_intercept, dependent_Y)
ssres = ss_res(predictions_with_intercept, dependent_Y)

rsq_with_intercept = 1 - (ssres / sstotal)

print('SStotal:', round(sstotal,4))
print('SSreg:', round(ssreg,4))
print('SSres:', round(ssres,4))
print('Coefficient of Determinacy: ', np.round(rsq_with_intercept,2))

# Residual analysis
residuals = dependent_Y - predictions_with_intercept
degrees_of_freedom = len(dependent_Y) - independent_X.shape[1]

residuals_standard_error = np.sqrt( (1/degrees_of_freedom) *  np.sum(residuals ** 2))
mean_independent_variable = np.mean(independent_X[:,1])
leverage = ( (1/len(independent_X)) 
             + (((independent_X[:,1] - mean_independent_variable)**2) 
                / np.sum((independent_X[:,1] - mean_independent_variable)**2)) )
standardized_rediduals = residuals / (residuals_standard_error * np.sqrt( 1 - leverage))

plt.clf()

sm.qqplot(standardized_rediduals, 
          stats.t,
          distargs=(degrees_of_freedom,), 
          line='q')

#plt.show()

print("Skewness of residuals: " , skew(standardized_rediduals))

plt.clf()

# Log Transformation
## Define masks
mask_dependent = dependent_Y != 0
mask_independent = independent_X[:,1] != 0
## Remove zero cases
dependent_Y_masked = dependent_Y[mask_dependent]
independent_X_masked = independent_X[:,1][mask_independent]
## transform
dependent_Y_masked = np.log(dependent_Y_masked)
independent_X_masked = np.log(independent_X_masked)

# Linear Regression
## add ones vector again
ones_vector = np.ones(len(independent_X_masked))
independent_X_masked = np.c_[ones_vector, independent_X_masked]
## perform regression
beta = np.linalg.lstsq(independent_X_masked, dependent_Y_masked, rcond=None)[0]
print('Estimated coefficients of log transformed regression:', beta)
predictions_with_intercept = dot(independent_X_masked, beta)
## visualize fitting
ax = sns.regplot(x=independent_X_masked[:,1], y=dependent_Y_masked, fit_reg=False, scatter_kws={'alpha':0.4})
ax.set(xlabel='percentage of people being divorced', ylabel='percentage of people in poverty', title='log transformed regression') 
plt.plot(independent_X_masked[:,1], predictions_with_intercept)


#plt.show()

# Model Fitting
## calculate rsq
sstotal = ss_total(dependent_Y_masked)
ssreg = ss_reg(predictions_with_intercept, dependent_Y_masked)
ssres = ss_res(predictions_with_intercept, dependent_Y_masked)
rsq_with_intercept = 1 - (ssres / sstotal)
print('SStotal:', round(sstotal,4))
print('SSreg:', round(ssreg,4))
print('SSres:', round(ssres,4))
print('correlation coeff with log:', np.corrcoef(dependent_Y_masked, independent_X_masked[:,1]))
print('Coefficient of Determinacy, model with log transformation', np.round(rsq_with_intercept,2))

# Residuals analysis
residuals = dependent_Y_masked - predictions_with_intercept
degrees_of_freedom = len(dependent_Y_masked) - independent_X_masked.shape[1]

residuals_standard_error = np.sqrt( (1/degrees_of_freedom) *  np.sum(residuals ** 2))
mean_independent_variable = np.mean(independent_X_masked[:,1])
leverage = ( (1/len(independent_X_masked)) 
             + (((independent_X_masked[:,1] - mean_independent_variable)**2) 
                / np.sum((independent_X_masked[:,1] - mean_independent_variable)**2)) )
standardized_rediduals = residuals / (residuals_standard_error * np.sqrt( 1 - leverage))
print("Skewness after log tranformation: " , skew(standardized_rediduals))

sm.qqplot(standardized_rediduals, 
          stats.t,
          distargs=(degrees_of_freedom,), 
          line='q')
plt.clf()
print('mean stan: ', np.mean(standardized_rediduals))
sns.regplot(x=predictions_with_intercept,
            y=standardized_rediduals, 
            lowess=True, 
            scatter_kws={'alpha':0.3}, 
            line_kws={"color":"r","alpha":0.4,"lw":2})
plt.plot(np.arange(len(standardized_rediduals)), [0]*len(standardized_rediduals), 'r-')
plt.xlim(0,predictions_with_intercept.max() + 0.1)
plt.xlabel('Fitted values')
plt.ylabel('Standardized Residuals')
plt.title('Standardized Residuals vs Fitted values')
plt.show()

# Calculate Coefficient standard errors
sigma_degrees_of_freedom = (independent_X.shape[0]-independent_X.shape[1])
sigma_sqr = np.sum(residuals**2)/sigma_degrees_of_freedom
variance_covmatrix = inv(dot(independent_X.T,independent_X)) * sigma_sqr
coeff_stde = np.diag(np.sqrt(variance_covmatrix))
print('Coefficients\' Standard Errors:',coeff_stde)

# Hypothesis testing
n = len(dependent_Y)
r_p = pearsonr(dependent_Y_masked, independent_X_masked[:,1])
print("n: ", n)
print("r: ", r_p[0])
#t = (r_p[0] * np.sqrt(n - 2)) / np.sqrt(1 - r_p[0]**2)
t = beta[1] / coeff_stde[1]
print("t: ", t)
print("p: ", r_p[1])

# confidence interval for slope
upper_bound = beta[1] + t * coeff_stde[1]
lower_bound = beta[1] - t * coeff_stde[1]
print('95 percent interval (slope): [', lower_bound, ',', upper_bound, ']')

# confidence interval for correlation coefficient
upper_bound = r_p[0] + t * (np.sqrt((1 - r_p[0]**2) / (n - 2)))
lower_bound = r_p[0] - t * (np.sqrt((1 - r_p[0]**2) / (n - 2)))
print('95 percent interval (correlation coeff): [', lower_bound, ',', upper_bound, ']')

# confidence interval for mean
y_values_for_specific_x = [4.57, 2.14, 3.32]
upper_bound = beta[1] * 4.41 + beta[0] + t * coeff_stde[1] * np.sqrt(1/n + ((4.41 - np.mean(independent_X))**2)/np.sum(list(map(lambda x: (x - np.mean(independent_X))**2,independent_X))))
lower_bound = (beta[1] * 4.41 + beta[0]) - t * coeff_stde[1] * np.sqrt(1/n + ((4.41 - np.mean(independent_X))**2)/np.sum(list(map(lambda x: (x - np.mean(independent_X))**2, independent_X))))
print('95 percent interval (y mean for given x): [', lower_bound, ',', upper_bound, ']')

# confidence interval for mean / single value
upper_bound = beta[1] * 4.41 + beta[0] + t * coeff_stde[1] * np.sqrt(1 + 1/n + ((4.41 - np.mean(independent_X))**2)/np.sum(list(map(lambda x: (x - np.mean(independent_X))**2,independent_X))))
lower_bound = beta[1] * 4.41 + beta[0] - t * coeff_stde[1] * np.sqrt(1 + 1/n + ((4.41 - np.mean(independent_X))**2)/np.sum(list(map(lambda x: (x - np.mean(independent_X))**2, independent_X))))
print('95 percent interval (random y for given x): [', lower_bound, ',', upper_bound, ']')