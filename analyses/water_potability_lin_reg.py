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

df = pd.read_csv('water_potability_cleaned_mean.csv')

solids_array = df['Solids'].to_numpy().reshape(-1,1)
sulfates_array = df['Sulfate'].to_numpy()

ax = sns.regplot(x=solids_array, y=sulfates_array, fit_reg=False, scatter_kws={'alpha':0.4})
ax.set(xlabel='Solids', ylabel='Sulfate', title='')

#plt.show()

# Linear Regression
## add ones vector
ones_vector = np.ones(len(solids_array))
solids_array = np.c_[ones_vector, solids_array]
## perform regression
beta = np.linalg.lstsq(solids_array, sulfates_array, rcond=None)[0]
print('Estimated coefficients:', beta)
predictions_with_intercept = dot(solids_array, beta) 
## visualize fitting
plt.plot(solids_array[:,1], predictions_with_intercept)

#plt.clf()
#plt.show()

# Model Fitting
## calculate rsq
sstotal = ss_total(sulfates_array)
ssreg = ss_reg(predictions_with_intercept, sulfates_array)
ssres = ss_res(predictions_with_intercept, sulfates_array)

rsq_with_intercept = 1 - (ssres / sstotal)

print('SStotal:', round(sstotal,4))
print('SSreg:', round(ssreg,4))
print('SSres:', round(ssres,4))
print('Coefficient of Determinacy: ', np.round(rsq_with_intercept,2))

# Residual analysis
residuals = sulfates_array - predictions_with_intercept
degrees_of_freedom = len(sulfates_array) - solids_array.shape[1]

residuals_standard_error = np.sqrt( (1/degrees_of_freedom) *  np.sum(residuals ** 2))
mean_independent_variable = np.mean(solids_array[:,1])
leverage = ( (1/len(solids_array)) 
             + (((solids_array[:,1] - mean_independent_variable)**2) 
                / np.sum((solids_array[:,1] - mean_independent_variable)**2)) )
standardized_rediduals = residuals / (residuals_standard_error * np.sqrt( 1 - leverage))

sm.qqplot(standardized_rediduals, 
          stats.t,
          distargs=(degrees_of_freedom,), 
          line='q')

#plt.show()

print("Skewness of residuals: " , skew(standardized_rediduals))

# Log Transformation
## Define masks
mask_sulfates = sulfates_array != 0
mask_solids = solids_array[:,1] != 0
## Remove zero cases
sulfates_array_masked = sulfates_array[mask_sulfates]
solids_array_masked = solids_array[:,1][mask_solids]
## transform
sulfates_array_masked = np.log(sulfates_array_masked)
solids_array_masked = np.log(solids_array_masked)

# Linear Regression
## add ones vector again
ones_vector = np.ones(len(solids_array_masked))
solids_array_masked = np.c_[ones_vector, solids_array_masked]
## perform regression
beta = np.linalg.lstsq(solids_array_masked, sulfates_array_masked, rcond=None)[0]
print('Estimated coefficients of log transformed regression:', beta)
predictions_with_intercept = dot(solids_array_masked, beta)
## visualize fitting
ax = sns.regplot(x=solids_array_masked[:,1], y=sulfates_array_masked, fit_reg=False, scatter_kws={'alpha':0.4})
ax.set(xlabel='Solids', ylabel='Sulfate', title='log transformed regression') 
plt.plot(solids_array_masked[:,1], predictions_with_intercept)

#plt.clf()
#plt.show()

# Model Fitting
## calculate rsq
sstotal = ss_total(sulfates_array_masked)
ssreg = ss_reg(predictions_with_intercept, sulfates_array_masked)
ssres = ss_res(predictions_with_intercept, sulfates_array_masked)
rsq_with_intercept = 1 - (ssres / sstotal)
print('SStotal:', round(sstotal,4))
print('SSreg:', round(ssreg,4))
print('SSres:', round(ssres,4))
print('Coefficient of Determinacy, model with log transformation', np.round(rsq_with_intercept,2))

# Residuals analysis
residuals = sulfates_array_masked - predictions_with_intercept
degrees_of_freedom = len(sulfates_array_masked) - solids_array_masked.shape[1]

residuals_standard_error = np.sqrt( (1/degrees_of_freedom) *  np.sum(residuals ** 2))
mean_independent_variable = np.mean(solids_array_masked[:,1])
leverage = ( (1/len(solids_array_masked)) 
             + (((solids_array_masked[:,1] - mean_independent_variable)**2) 
                / np.sum((solids_array_masked[:,1] - mean_independent_variable)**2)) )
standardized_rediduals = residuals / (residuals_standard_error * np.sqrt( 1 - leverage))
print("Skewness after log tranformation: " , skew(standardized_rediduals))

sm.qqplot(standardized_rediduals, 
          stats.t,
          distargs=(degrees_of_freedom,), 
          line='q')
#plt.clf()
#plt.show()

# Hypothesis testing
n = len(sulfates_array)
r_p = pearsonr(sulfates_array, solids_array[:,1])
print("n: ", n)
print("r: ", r_p[0])
t = (r_p[0] * np.sqrt(n - 2)) / np.sqrt(1 - r_p[0]**2)
print("t: ", t)
print("p: ", r_p[1])

# Calculate Coefficient standard errors
sigma_degrees_of_freedom = (solids_array.shape[0]-solids_array.shape[1])
sigma_sqr = np.sum(residuals**2)/sigma_degrees_of_freedom
variance_covmatrix = inv(dot(solids_array.T,solids_array)) * sigma_sqr
coeff_stde = np.diag(np.sqrt(variance_covmatrix))
print('Coefficients\' Standard Errors:',coeff_stde)

# confidence interval for slope
upper_bound = beta[1] + t * coeff_stde[1]
lower_bound = beta[1] - t * coeff_stde[1]
print('95 percent interval (slope): [', lower_bound, ',', upper_bound, ']')

# confidence interval for correlation coefficient
upper_bound = r_p[0] + t * (np.sqrt((1 - r_p[0]**2) / (n - 2)))
lower_bound = r_p[0] - t * (np.sqrt((1 - r_p[0]**2) / (n - 2)))
print('95 percent interval (correlation coeff): [', lower_bound, ',', upper_bound, ']')

# confidence interval for mean / single value
upper_bound = sulfates_array[2314] + t * np.std(solids_array[:,1]) / np.sqrt(n)
lower_bound = sulfates_array[2314] - t * np.std(solids_array[:,1]) / np.sqrt(n)
print('95 percent interval (y mean of fixed x): [', lower_bound, ',', upper_bound, ']')