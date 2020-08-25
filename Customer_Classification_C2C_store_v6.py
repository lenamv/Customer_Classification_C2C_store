#!/usr/bin/env python
# coding: utf-8

# # Market Segmentation for a French (C2C) Fashion Store
# Market segmentation for a french (C2C) fashion store using cluster analysis to achieve more effective customer marketing via personalization.
# The data source is 'data.world' portal https://data.world/jfreex/e-commerce-users-of-a-french-c2c-fashion-store.  

# ### Importing packages, loading the data
# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
import collections

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import cluster, tree, decomposition
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Disable zero division warnings
np.errstate(divide='ignore')

# Set matplotlib options
color = '#1F77B4'
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)

# Download the dataset
# Source https://data.world/jfreex/e-commerce-users-of-a-french-c2c-fashion-store
fl = '2. Prepared Data/6M-0K-99K.users.dataset.public.csv'
data = pd.read_csv(fl)

# Preview data
data.head()

# ## 1. Data Preprocessing
# ### 1.1. Exploratory Data Analysis & Data Cleaning

# Get base information on the dataset
data.columns

# Drop columns with information that will not be used or is duplicated 
drop_columns = ['civilityGenderId', 'civilityTitle', 'seniorityAsMonths', 'seniorityAsYears','countryCode']
data.drop(columns=drop_columns, inplace=True)
data.info()

# Get the dimensionality of the dataset
print(data.shape)


# Check that user identification numbers are unique
users = len(data['identifierHash'].unique().tolist())
print(f'Duplicated users: {users-data.shape[0]}')

# Chech type of users is 'user' 
user_types = (data['type']=='user').sum() - data.shape[0]

# Drop 'type' column if type of all users is 'user' 
if user_types==0:
    data.drop(columns='type', inplace=True)
    
# Drop rows if type of users is different from 'user'
# Then drop 'type' column
else:
    other_types = data['type']!='user'
    data.drop(data[other_types].index, inplace=True)
    data.drop(columns='type', inplace=True)

# Check duplicated rows
print(f'Duplicated rows : {data.duplicated().sum()}')
# Find missing values 
print(f'Missing values: {data.isnull().sum().sum()}')

# Split up numeric, categorical, and boolean columns 
print('Columns')
numeric_cols = data.select_dtypes([np.int64,np.float64]).columns.tolist()
print (f'Numeric: {len(numeric_cols)} \n{numeric_cols}')

categorical_cols = data.select_dtypes([np.object]).columns.tolist()
print (f'Categorical: {len(categorical_cols)} \n{categorical_cols}')

booleans_cols = data.select_dtypes([np.bool]).columns.tolist()
print (f'Boolean: {len(booleans_cols)} \n{booleans_cols}')

# Filter data of active users, who sold, bought, wished or liked products

###################################### FILTER ################################################

# List of columns related to user activity in selling of products
products = ['socialProductsLiked', 'productsListed', 'productsSold', 'productsWished', 'productsBought']
# Calculate sum across rows
data['productSum'] = data[products].sum(axis=1)

# Filter data of user who have any activity (sum > 0)
data = data[data['productSum'] > 0]

# Drop utility 'productSum' column
data.drop(columns='productSum', inplace=True)

# Reset index
data.reset_index(drop=True, inplace=True)

# Preview data
data.head()

# Get the dimensionality of the dataset
print(data.shape)

# Get statistics for numeric columns
data.drop(columns='identifierHash').describe() 

# Plot the distribution of numerical columns

# Create list of numerical columns for plots
# Copy the list of numerical columns
numeric_cols_plot = numeric_cols.copy()
# Remove the 'identifierHash' column
numeric_cols_plot.remove('identifierHash')

# Translate the list of countries into English 
data['country'] = data['country'].replace({
     'Royaume-Uni':'UK', 'Danemark':'Denmark','Etats-Unis':'US', 'Allemagne':'Germany', 
     'Suisse':'Swiss','Suède':'Sweden','Australie':'Australia','Italie':'Italy', 'Espagne':'Spain', 
     'Finlande':'Finland','Belgique':'Belgium','Pays-Bas':'Netherlands', 'Autriche':'Austria', 
    'Russie':'Russia', 'Bulgarie':'Bulgaria', 'Chine':'China', 'Irlande': 'Ireland', 'Roumanie':'Romania'})

# Group users by country
countries=pd.DataFrame(data['country'].value_counts()).sort_values(by='country',ascending=False)
countries.rename(columns={'country':'users'}, inplace=True)
# Calculate % of total users by country
countries['users_%_total'] = round(countries['users']/countries['users'].sum(),3)*100

# Number of users by country
users_20 = countries.head(20)
users_20

# ### 1.2. Feature Engineering

# Define countries as 'Other', if % of users is less then 1%
users_perc_lim = 1
other_countries = countries[countries['users_%_total']<users_perc_lim].index.tolist()
print(f'Other countries: {len(other_countries)}')
#print(other_countries)
data['country_short_list']=data['country'].apply(lambda x: 'Other' if x in other_countries else x)
countries_list = data['country_short_list'].unique().tolist()
print(f'Countries short list: {countries_list}')

# Save the initial data to use with results of clustering
data_origin = data.copy()
data_origin.head()

# Change the boolean column type to integer
data[booleans_cols]=data[booleans_cols].astype(int)

# Duplicate columns
data['language_c']=data['language']
# Convert categorical variable into dummy/indicator variables
cat_features =['language', 'gender', 'country_short_list']
data = pd.get_dummies(data, columns=cat_features)

# Rename columns
data.rename(columns={'language_c': 'language'}, inplace=True)

# Preview data
data.drop(columns=['identifierHash']).head()

# Check that 'hasAnyApp' is 'hasAndroidApp' or 'hasIosApp'
(data['hasAndroidApp']+data['hasIosApp']-data['hasAnyApp']).isnull().sum()

# Save the preprocessed data to a csv file
file = '2. Prepared Data/classification_e-commerce_preprocessed_data_py.csv'
data.to_csv(file, index = False)
data.shape


# ## 2. Clustering
# 
# ### 2.1. Features 

features = ['productsListed', 'productsSold', 'productsBought', 
            'productsWished','socialProductsLiked','socialNbFollowers','socialNbFollows',
            'daysSinceLastLogin']

print(f'Features: {features}')

# Set up the data
df_scl=data[features].copy()

# Scale the data using RobustScaler as it is based on percentiles and not influenced by outliers
df_scl[features] = RobustScaler().fit_transform(df_scl)

# Principal component analysis for visualization
pca = PCA(n_components=2, whiten=True)
df_pca = pca.fit_transform(df_scl[features])

df_scl['x']=df_pca[:,0]
df_scl['y']=df_pca[:,1]

# Pick out the most significant outliers
outliers = df_scl[df_scl['x']>7]
data.loc[outliers.index]

# Drop the selected outliers
droped_outliers = data.loc[outliers.index]

# Save the clustered data to a csv file
file = '4. Analysis/droped_outliers_classification_e-commerce_clusters_py.csv'
droped_outliers.to_csv(file, index = False)

data.drop(outliers.index, inplace=True)
print(data.shape)

# Set up the data
df = data[features].copy()

# ### 2.4. Final Model
param_kmeans_n_clusters = 5

# KMeans and Agglomerative Clustering give almost the same silhouette soefficient and distribution of samples for 5 clusters.
# KMeans clustering shows the highest silhouette soefficient, and clusters are a bit more clearly separated on the scatter plot. 
# So that KMeans(n_clusters=5, max_iter=500) is considered as the final model.

# Set up the data
df_f = data[features].copy()

# Scale the data using RobustScaler as it is based on percentiles and not influenced by outliers
df_scl = RobustScaler().fit_transform(df_f)

# Principal component analysis for visualization
pca = PCA(n_components=2, whiten=True)
df_pca = pca.fit_transform(df_scl)

df['x'] = df_pca[:,0]
df['y'] = df_pca[:,1]

# Scaled data
X = df_scl
#---------------------------------- Kmeans -------------------------------------------------
kmeans = KMeans(n_clusters=param_kmeans_n_clusters, max_iter=500, random_state=None).fit(X)

print('Model:')
print(kmeans)

labels = kmeans.labels_
n_clusters = len(set(labels))
df['kmeans']=labels

# Plot the result
plt.figure(figsize=(8,6))
plt.scatter(df_pca[:,0],df_pca[:,1], c=labels, edgecolors='k', s=84)

plt.title(f'KMeans Clustering, n_clusters={n_clusters}')
plt.show()

# Silhouette Score
sil_coeff = silhouette_score(X, labels, metric='euclidean')
print(f'n_clusters={n_clusters}, Silhouette Coefficient: {sil_coeff}')
# Count the occurrence of samples in an clusters
clusters = collections.Counter(labels)
print(f'Clusters: {clusters}')

# Add cluster labels and PCA components to the original DataFrame
result = data_origin.merge(df[['x', 'y', 'kmeans']],how='outer', left_index=True,right_index=True)
print(f'Result: {result.shape}')

# Save the clustered data to a csv file
file = '5. Insights/classification_e-commerce_clusters_py.csv'
result.to_csv(file, index = False)

# ## 3. Profile and Inspect 5 clusters

# Clusters labels from the final model
model_clusters = 'kmeans'
# Count a number of samples in clusters
result[model_clusters].value_counts()


# Create a DataFrame with features and labels of the clusters
clusters = result[features + [model_clusters]].groupby(model_clusters).mean()

# Count a number of samples in clusters and convert result to a DataFrame
labels_sum = pd.DataFrame(result[model_clusters].value_counts())

# Add sum of clusters as column to the clusters DataFrame
clusters_sum = pd.concat([clusters, labels_sum], axis=1)

# Find mean for columns, convert the result to a DataFrame and transpose it  
total_mean =pd.DataFrame(result[features ].mean(), columns=['mean']).T

# Add means as a row to the DataFrame with cluster data
print('CLUSTERS')
clusters_df = pd.concat([clusters_sum, total_mean])
print(clusters_df)

