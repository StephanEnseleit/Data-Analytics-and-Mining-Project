import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('./data/communities_processed.csv')

print(df.head())
df_violent_crimes = df['ViolentCrimesPerPop']
df = df.drop(columns=['ViolentCrimesPerPop'])
# Scale data before applying PCA
scaling=StandardScaler()

# Use fit and transform method
scaling.fit(df)
Scaled_data=scaling.transform(df)

# Set the n_components=3
principal=PCA(n_components=3)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)
principalDf = pd.DataFrame(data = x, columns = ['pc1', 'pc2', "pc3"])
print(principalDf)

# import relevant libraries for 3d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
 
# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')
 
# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
axis.scatter(x[:, 0], x[:, 1], x[:, 2], c=df_violent_crimes, cmap='hot')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)
plt.show()