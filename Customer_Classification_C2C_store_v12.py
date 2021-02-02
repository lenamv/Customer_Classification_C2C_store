#!/usr/bin/env python
# coding: utf-8

# # Market Segmentation for a French (C2C) Fashion Store
# Market segmentation for a french (C2C) fashion store using cluster analysis to achieve more effective customer marketing via personalization.     
# The goal of the project is to find out a ratio of types of customers who are sellers to support decision making in customer marketing.    
# The data source is 'data.world' portal https://data.world/jfreex/e-commerce-users-of-a-french-c2c-fashion-store.    
# The focus of the project is customers who are active users of the platform in selling field. To be consider as an active user, they should have at least one sold product.

# ### Importing packages, loading the data
# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import collections

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn import cluster, tree, decomposition
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Disable zero division warnings
np.errstate(divide='ignore')

# Download the dataset
# Source https://data.world/jfreex/e-commerce-users-of-a-french-c2c-fashion-store
fl = '2. Prepared Data/6M-0K-99K.users.dataset.public.csv'
data = pd.read_csv(fl)

# ## 1. Data Preprocessing
# ### 1.1. Exploratory Data Analysis & Data Cleaning

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

# Filter data of active users, who has sold products
###################################### FILTER ################################################

# List of columns related to user activity in selling of products
#products = ['socialProductsLiked', 'productsListed', 'productsSold', 'productsWished', 'productsBought']
#products = ['productsSold', 'productsListed']
products = ['productsSold']
# Calculate sum across rows
data['productSum'] = data[products].sum(axis=1)

# Filter data of user who have any activity (sum > 0)
data = data[data['productSum'] > 0]
###################################### END FILTER ################################################

# Drop utility 'productSum' column
data.drop(columns=['productSum', 'identifierHash'], inplace=True)

# Reset index
data.reset_index(drop=True, inplace=True)

# Get the dimensionality of the dataset
print(data.shape)

# Split up numeric, categorical, and boolean columns 
print('Columns')
numeric_cols = data.select_dtypes([np.int64,np.float64]).columns.tolist()
print (f'Numeric: {len(numeric_cols)} \n{numeric_cols}')

categorical_cols = data.select_dtypes([np.object]).columns.tolist()
print (f'Categorical: {len(categorical_cols)} \n{categorical_cols}')

booleans_cols = data.select_dtypes([np.bool]).columns.tolist()
print (f'Boolean: {len(booleans_cols)} \n{booleans_cols}')

# Count number of countries
countries_number = len(data['country'].unique().tolist())
print(f'Countries: {countries_number}')

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


# ### 1.2. Data preprocessing, Feature Engineering
# Define countries as 'Other', if % of users is less then 1%
users_perc_lim = 1
other_countries = countries[countries['users_%_total']<users_perc_lim].index.tolist()
print(f'Other countries: {len(other_countries)}')
#print(other_countries)
data['country_short_list']=data['country'].apply(lambda x: 'Other' if x in other_countries else x)
countries_list = data['country_short_list'].unique().tolist()
print(f'Countries short list: {countries_list}')

# Drop extreme outliers
filter_outliers = (data['socialNbFollows']>5000) | (data['socialNbFollowers']>500) | (data['socialProductsLiked']>10000)
data.drop(data[filter_outliers].index, inplace=True)

# Save the outliers to a csv file
file = '4. Analysis/droped_outliers_classification_e-commerce_clusters_py.csv'
data.to_csv(file, index = False)
print(data.shape)

# Feature engineering
data['social']= data['socialNbFollowers'] + data['socialNbFollows'] 
data['products'] = data['socialProductsLiked'] + data['productsWished']

# Save the initial data to use with results of clustering
data_origin = data.copy()

# Transform predictor variables using Yeo-Johnson transformation
columns = ['socialNbFollowers', 'socialNbFollows', 'socialProductsLiked', 'productsListed', 
           'productsSold', 'productsWished', 'productsBought', 'daysSinceLastLogin','social', 'products', 
           'productsPassRate', 'seniority']

qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')
data[columns] = qt.fit_transform(data[columns])

# Change the boolean column type to integer
data[booleans_cols]=data[booleans_cols].astype(int)

# Duplicate columns
data['language_c']=data['language']
# Convert categorical variable into dummy/indicator variables
cat_features =['language', 'gender', 'country_short_list']
data = pd.get_dummies(data, columns=cat_features, drop_first=True)

# Rename columns
data.rename(columns={'language_c': 'language'}, inplace=True)

# Check that 'hasAnyApp' is 'hasAndroidApp' or 'hasIosApp'
(data['hasAndroidApp']+data['hasIosApp']-data['hasAnyApp']).isnull().sum()

# Save the preprocessed data to a csv file
file = '2. Prepared Data/classification_e-commerce_preprocessed_data_py.csv'
data.to_csv(file, index = False)
data.shape 


# ## 2. Clustering
# ### 2.1. Features 
# Select features
features = ['socialNbFollowers', 'socialNbFollows', 'socialProductsLiked', 'productsListed', 'productsSold',
       'productsPassRate', 'productsWished', 'productsBought', 'social', 'products']
print(f'Features: \n {features}')


# ### 2.3. Model Selection
# Select the best parameter
# Select the best parameter when n_clusters > 2
param_kmeans_n_clusters = 4
param_spectral_n_clusters = 4
param_param_aggl_n_clusters = 4

print(f'Selected parameters:')
print(f'Kmeans parameters: n_clusters = {param_kmeans_n_clusters}')
print(f'Spectral Clustering parameters: n_clusters = {param_spectral_n_clusters}')
print(f'Agglomerative Clustering parameters: n_clusters = {param_param_aggl_n_clusters}')

# Set up the data
df = data[features].copy()

# Scale the data 
df_scl = StandardScaler().fit_transform(df)

# Principal component analysis for visualization
pca = PCA(n_components=2, whiten=True)
df_pca = pca.fit_transform(df_scl)

df['x'] = df_pca[:,0]
df['y'] = df_pca[:,1]

# Scaled data
X = df_scl
#---------------------------------- Kmeans -------------------------------------------------
kmeans = KMeans(n_clusters=param_kmeans_n_clusters, max_iter=500).fit(X)
labels = kmeans.labels_
n_clusters = len(set(labels))
df['kmeans']=labels

print(kmeans)
# Silhouette Score
sil_coeff = silhouette_score(X, labels, metric='euclidean')
print(f'n_clusters={n_clusters}, Silhouette Coefficient: {sil_coeff}')
# Count the occurrence of samples in clusters
clusters = collections.Counter(labels)
print(f'Clusters: {clusters}')

#---------------------------------- Spectral Clustering -------------------------------------------------

spectral = SpectralClustering(n_clusters=param_spectral_n_clusters, assign_labels='discretize', random_state=0).fit(X)
labels = spectral.labels_
n_clusters = len(set(labels))
df['spectral']=labels

print(spectral)
# Silhouette Score
sil_coeff = silhouette_score(X, labels, metric='euclidean')
print(f'n_clusters={n_clusters}, Silhouette Coefficient: {sil_coeff}')
# Count the occurrence of samples in clusters
clusters = collections.Counter(labels)
print(f'Clusters: {clusters}')


# ---------------------------------- Agglomerative Clustering -------------------------------------
aggl = AgglomerativeClustering(n_clusters=param_param_aggl_n_clusters, affinity='euclidean', linkage='ward').fit(X)

labels = aggl.labels_
n_clusters = len(set(labels))
df['aggl']=labels

print(aggl)
# Silhouette Score
sil_coeff = silhouette_score(X, labels, metric='euclidean')
print(f'n_clusters={n_clusters}, Silhouette Coefficient: {sil_coeff}')
# Count the occurrence of samples in clusters
clusters = collections.Counter(labels)
print(f'Clusters: {clusters}')


# Save the results of clustering to a csv file
print(f'data_origin: {data_origin.shape}')
print(f'df: {df.shape}')

# Add cluster labels and PCA components to the original DataFrame
result = data_origin[features+ ['country', 'daysSinceLastLogin']].join(df[['x', 'y', 'kmeans', 'spectral', 'aggl']])
print(f'result: {result.shape}')


# ### 2.4. The Final Model

# Set up the data
df_f = data[features].copy()

# Scale the data 
df_scl = StandardScaler().fit_transform(df_f)

# Principal component analysis for visualization
pca = PCA(n_components=2, whiten=True)
df_pca = pca.fit_transform(df_scl)

df['x'] = df_pca[:,0]
df['y'] = df_pca[:,1]

# Scaled data
X = df_scl

#---------------------------------- Kmeans -------------------------------------------------
aggl = AgglomerativeClustering(n_clusters=param_param_aggl_n_clusters, affinity='euclidean', linkage='ward').fit(X)
labels = aggl.labels_
n_clusters = len(set(labels))
df['aggl']=labels

print('Final Model:')
print(aggl)

# Silhouette Score
sil_coeff = silhouette_score(X, labels, metric='euclidean')
print(f'n_clusters={n_clusters}, Silhouette Coefficient: {sil_coeff}')
# Count the occurrence of samples in an clusters
clusters = collections.Counter(labels)
print(f'Clusters: {clusters}')


# ## 3. Profile of clusters
# The final model
model_clusters = 'aggl'

# Clusters labels from the final model
result[model_clusters]=labels

# Count a number of samples in clusters
result[model_clusters].value_counts()

# Save the clustered data to a csv file
file = '5. Insights/classification_e-commerce_clusters_py.csv'
result.to_csv(file, index = False)

