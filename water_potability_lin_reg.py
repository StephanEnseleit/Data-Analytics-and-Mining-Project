import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.linalg import inv
from numpy.linalg import det
from numpy import dot
import statsmodels.api as sm
sns.set_style("darkgrid")

df = pd.read_csv('water_potability_cleaned_mean.csv')

solids_array = df['Solids'].to_numpy().reshape(-1,1)
sulfates_array = df['Sulfate'].to_numpy()

ax = sns.regplot(x=solids_array, y=sulfates_array, fit_reg=True, scatter_kws={'alpha':0.4})
ax.set(xlabel='Solids', ylabel='Sulfate', title='Linear Regression')

ones_vector = np.ones(len(solids_array))
solids_array = np.c_[ones_vector, solids_array]
beta = np.linalg.lstsq(solids_array, sulfates_array)[0]
print('Estimated coefficients:', beta)

predictions_with_intercept = dot(solids_array, beta) 
plt.plot(solids_array[:,1], predictions_with_intercept)
plt.show()

