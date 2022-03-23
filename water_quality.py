import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('water_potability.csv')
df['ph'].replace(np.nan, df['ph'].mean(), inplace=True)
#df.dropna(subset=['ph'], inplace=True)
df['Hardness'].replace(np.nan, df['Hardness'].mean(), inplace=True)
#df.dropna(subset=['Hardness'], inplace=True)
df['Solids'].replace(np.nan, df['Solids'].mean(), inplace=True)
#df.dropna(subset=['Solids'], inplace=True)
df['Chloramines'].replace(np.nan, df['Chloramines'].mean(), inplace=True)
#df.dropna(subset=['Chloramines'], inplace=True)
df['Sulfate'].replace(np.nan, df['Sulfate'].mean(), inplace=True)
#df.dropna(subset=['Sulfate'], inplace=True)
df['Conductivity'].replace(np.nan, df['Conductivity'].mean(), inplace=True)
#df.dropna(subset=['Conductivity'], inplace=True)
df['Organic_carbon'].replace(np.nan, df['Organic_carbon'].mean(), inplace=True)
#df.dropna(subset=['Organic_carbon'], inplace=True)
df['Trihalomethanes'].replace(np.nan, df['Trihalomethanes'].mean(), inplace=True)
#df.dropna(subset=['Trihalomethanes'], inplace=True)
df['Turbidity'].replace(np.nan, df['Turbidity'].mean(), inplace=True)
#df.dropna(subset=['Turbidity'], inplace=True)

df.to_csv('water_potability_cleaned_mean.csv')