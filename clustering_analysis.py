# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:43:36 2024

@author: erikleblansch
"""
import pandas as pd
import os

# Define the main path (change '<MAIN_PATH>' to your actual main path)
main_path = '<MAIN_PATH>'
os.chdir(main_path)

# Define the folder path containing the data files
folder_path = os.path.join(main_path, 'Data_KNMI')

# List all files in the folder
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Initialize an empty list to hold dataframes
dataframes = []

# Loop through each file and read the data, skipping the first 31 rows
for file in files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, skiprows=31, header=None, dtype=str)
    dataframes.append(df)

# Combine all dataframes into a single dataframe
KNMI_data = pd.concat(dataframes, ignore_index=True)

# Set column names (adjust based on actual data structure if necessary)
column_names = [
    'STN', 'YYYYMMDD', 'HH', 'Windrichting (in graden)', 'Uurgemiddelde windsnelheid (in 0.1 m/s)', 'Windsnelheid (in 0.1 m/s)', 
    'Hoogste windstoot (in 0.1 m/s)', 'Temperatuur (in 0.1 graden Celsius)', 'Minimumtemperatuur (in 0.1 graden Celsius)', 
    'Dauwpuntstemperatuur (in 0.1 graden Celsius)', 'Duur van de zonneschijn (in 0.1 uren)', 'Globale straling (in J/cm2)', 
    'Duur van de neerslag (in 0.1 uur)', 'Uursom van de neerslag (in 0.1 mm)', 'Luchtdruk (in 0.1 hPa)', 
    'Horizontaal zicht tijdens de waarneming', 'Bewolking (bedekkingsgraad van de bovenlucht in achtsten)', 
    'Relatieve vochtigheid (in procenten)', 'Weercode (00-99)', 'Weercode indicator', 'Mist', 'Regen', 
    'Sneeuw', 'Onweer', 'IJsvorming'
]
KNMI_data.columns = column_names

# Convert numeric columns to appropriate data types
numeric_columns = [
    'Windrichting (in graden)', 'Uurgemiddelde windsnelheid (in 0.1 m/s)', 'Windsnelheid (in 0.1 m/s)', 
    'Hoogste windstoot (in 0.1 m/s)', 'Temperatuur (in 0.1 graden Celsius)', 'Minimumtemperatuur (in 0.1 graden Celsius)', 
    'Dauwpuntstemperatuur (in 0.1 graden Celsius)', 'Duur van de zonneschijn (in 0.1 uren)', 'Globale straling (in J/cm2)', 
    'Duur van de neerslag (in 0.1 uur)', 'Uursom van de neerslag (in 0.1 mm)', 'Luchtdruk (in 0.1 hPa)', 
    'Horizontaal zicht tijdens de waarneming', 'Bewolking (bedekkingsgraad van de bovenlucht in achtsten)', 
    'Relatieve vochtigheid (in procenten)', 'Weercode (00-99)', 'Weercode indicator', 'Mist', 'Regen', 
    'Sneeuw', 'Onweer', 'IJsvorming'
]
KNMI_data[numeric_columns] = KNMI_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Prepare date and hour columns
KNMI_data['YYYYMMDD'] = KNMI_data['YYYYMMDD'].str.strip()
KNMI_data['HH'] = KNMI_data['HH'].str.strip().str.zfill(2)

# Filter out rows with invalid HH values
valid_hours = [f'{i:02}' for i in range(24)]
KNMI_data = KNMI_data[KNMI_data['HH'].isin(valid_hours)]

# Create a datetime index
KNMI_data['YYYYMMDD_HH'] = pd.to_datetime(KNMI_data['YYYYMMDD'] + KNMI_data['HH'], format='%Y%m%d%H', errors='coerce')
KNMI_data.set_index('YYYYMMDD_HH', inplace=True)

# Define columns to drop
columns_to_drop = ['YYYYMMDD', 'HH']  # 'STN' removed from the list

# Drop columns if they exist in DataFrame
for column in columns_to_drop:
    if column in KNMI_data.columns:
        KNMI_data.drop(column, axis=1, inplace=True)

# Save the cleaned data
KNMI_data.to_csv('KNMI_data.csv')

# Print the first few rows to confirm
print(KNMI_data.head())

#%%

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Set the main path and change the working directory
main_path = '<MAIN_PATH>'
os.chdir(main_path)

# Load the DataFrame
weather_table = pd.read_csv('KNMI_data.csv')
weather_table.replace([np.inf, -np.inf], np.nan, inplace=True)
weather_table.dropna(inplace=True)

# Detect numeric features for clustering
numeric_features = weather_table.select_dtypes(include=[np.number]).columns.tolist()
data = weather_table[numeric_features]

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Function to optimize KMeans clustering
def optimize_clusters_kmeans(X, max_clusters=30):
    best_score = -1
    best_k = 2
    best_kmeans = None
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans
    return best_k, best_kmeans, best_score, silhouette_scores

optimal_k, best_kmeans_model, best_silhouette_score_kmeans, kmeans_silhouettes = optimize_clusters_kmeans(data_scaled)
weather_table['Cluster_KMeans'] = best_kmeans_model.labels_

# Number of clusters in KMeans
num_clusters_kmeans = len(np.unique(best_kmeans_model.labels_))

# Finding the optimal eps value using NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(data_scaled)
distances, indices = neighbors_fit.kneighbors(data_scaled)

# Sort the distances
distances = np.sort(distances, axis=0)
distances = distances[:, 1]  # Take the distance to the closest point
plt.plot(distances[::-1])  # Reverse to sort from largest to smallest
plt.title("K-Nearest Neighbors Distance")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()

# Prompt user to input the optimal eps observed from the plot
eps_value = float(input("Enter the estimated eps value from the plot: "))

# Continue with DBSCAN clustering using the optimized eps value
dbscan = DBSCAN(eps=eps_value, min_samples=3)
dbscan_labels = dbscan.fit_predict(data_scaled)
silhouette_score_dbscan = silhouette_score(data_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
weather_table['Cluster_DBSCAN'] = dbscan_labels

# Number of clusters in DBSCAN
num_clusters_dbscan = len(np.unique(dbscan_labels))

# Agglomerative Clustering with different linkages
linkages = ['ward', 'complete', 'average']
linkage_results = {}
num_clusters_agg = {}
for linkage_type in linkages:
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage=linkage_type)
    agg_labels = agg_clustering.fit_predict(data_scaled)
    silhouette = silhouette_score(data_scaled, agg_labels)
    linkage_results[linkage_type] = silhouette
    weather_table[f'Cluster_Agglomerative_{linkage_type}'] = agg_labels
    num_clusters_agg[linkage_type] = len(np.unique(agg_labels))

# Gaussian Mixture Models
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_labels = gmm.fit_predict(data_scaled)
gmm_silhouette_score = silhouette_score(data_scaled, gmm_labels)
weather_table['Cluster_GMM'] = gmm_labels

# Number of clusters in GMM
num_clusters_gmm = len(np.unique(gmm_labels))

# Create a comparison table of clustering methods
comparison_table = pd.DataFrame({
    'Method': ['KMeans', 'DBSCAN', 'Agglomerative (Ward)', 'Agglomerative (Complete)', 'Agglomerative (Average)', 'Gaussian Mixture'],
    'Silhouette Score': [best_silhouette_score_kmeans, silhouette_score_dbscan] + [linkage_results[l] for l in linkages] + [gmm_silhouette_score],
    'Number of Clusters': [num_clusters_kmeans, num_clusters_dbscan] + [num_clusters_agg[l] for l in linkages] + [num_clusters_gmm]
})

print("\nComparison of Clustering Methods:")
print(comparison_table)

# Save the updated DataFrame with cluster labels
weather_table.to_csv('weather_clusters.csv')

#%%

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Set the main path and change the working directory
main_path = '<MAIN_PATH>'
os.chdir(main_path)

# Load the DataFrame
weather_table = pd.read_csv('weather_clusters.csv')
weather_table.replace([np.inf, -np.inf], np.nan, inplace=True)
weather_table.dropna(inplace=True)

# Detect numeric features for clustering
numeric_features = weather_table.select_dtypes(include=[np.number]).columns.tolist()
data = weather_table[numeric_features]

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)
weather_table['pca-one'] = pca_result[:, 0]
weather_table['pca-two'] = pca_result[:, 1]

# t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_scaled)
weather_table['tsne-one'] = tsne_results[:, 0]
weather_table['tsne-two'] = tsne_results[:, 1]

# UMAP for dimensionality reduction
reducer = umap.UMAP()
umap_results = reducer.fit_transform(data_scaled)
weather_table['umap-one'] = umap_results[:, 0]
weather_table['umap-two'] = umap_results[:, 1]

# Visualization of PCA, t-SNE, and UMAP for all methods
methods = ['Cluster_KMeans', 'Cluster_DBSCAN', 'Cluster_Agglomerative_ward', 'Cluster_Agglomerative_complete', 'Cluster_Agglomerative_average', 'Cluster_GMM']
for method in methods:
    plt.figure(figsize=(24, 7))
    for i, col in enumerate(['pca', 'tsne', 'umap']):
        plt.subplot(1, 3, i+1)
        sns.scatterplot(x=f"{col}-one", y=f"{col}-two", hue=method, palette=sns.color_palette("hsv", len(set(weather_table[method]))), data=weather_table, legend="full", alpha=0.3)
        plt.title(f'{col.upper()} projection of the {method}', fontsize=15)
    plt.show()

# Generate min and max values table for each cluster method
cluster_stats = {}
for method in methods:
    grouped = weather_table.groupby(method)[numeric_features].agg(['min', 'max'])
    cluster_stats[method] = grouped

# Print the statistics for all clustering methods
for method in methods:
    print(f"Statistics for {method} Clusters:")
    print(cluster_stats[method])
    print("\n")  # Adding a newline for better readability between methods

#%%
import pandas as pd

cluster_stats = {}
cluster_summary = {}

for method in methods:
    # Grouping by cluster method and aggregating min, max, and mean
    grouped = weather_table.groupby(method)[numeric_features].agg(['min', 'max', 'mean'])
    
    # Drop unnecessary columns if present
    grouped.drop(columns=['Unnamed: 0', 'STN'], errors='ignore', inplace=True)

    # Calculate rankings for specific metrics
    grouped['rank_max_wind_gust'] = grouped[('Hoogste windstoot (in 0.1 m/s)', 'max')].rank(ascending=True)
    grouped['rank_max_temp'] = grouped[('Temperatuur (in 0.1 graden Celsius)', 'max')].rank(ascending=True)
    grouped['rank_min_temp'] = grouped[('Temperatuur (in 0.1 graden Celsius)', 'min')].rank(ascending=False)
    grouped['rank_max_precip'] = grouped[('Uursom van de neerslag (in 0.1 mm)', 'max')].rank(ascending=True)
    grouped['rank_min_pressure'] = grouped[('Luchtdruk (in 0.1 hPa)', 'min')].rank(ascending=False)
    grouped['rank_min_visibility'] = grouped[('Horizontaal zicht tijdens de waarneming', 'min')].rank(ascending=False)

    # Calculate extremity score as the sum of all ranks
    grouped['extremity_score'] = grouped[['rank_max_wind_gust', 'rank_max_temp', 'rank_min_temp', 'rank_max_precip', 'rank_min_pressure', 'rank_min_visibility']].sum(axis=1)

    # Rank clusters based on the extremity score
    grouped['Weather Index'] = grouped['extremity_score'].rank(ascending=True)

    # Sort by 'Weather Index'
    cluster_stats[method] = grouped.sort_values(by='Weather Index')

    # Prepare summary data with correct column handling
    summary_cols = [
        ('Hoogste windstoot (in 0.1 m/s)', 'max'),
        ('Temperatuur (in 0.1 graden Celsius)', 'mean'),
        ('Uursom van de neerslag (in 0.1 mm)', 'max'),
        ('Luchtdruk (in 0.1 hPa)', 'min'),
        ('Horizontaal zicht tijdens de waarneming', 'min')
    ]
    summary = grouped.loc[:, summary_cols].copy()
    summary.columns = ['max Wind Gust', 'mean Temperature', 'max Precipitation', 'min Pressure', 'min Visibility']
    summary['Weather Index'] = grouped['Weather Index']

    cluster_summary[method] = summary.sort_values(by 'Weather Index')

# Adjust units by dividing specific columns by 10
for method, summary in cluster_summary.items():
    summary[['max Wind Gust', 'mean Temperature', 'max Precipitation']] /= 10

    # Print summary for each clustering method
    print(f"Updated Summary for {method} Clusters:")
    print(summary)
    print("\n")
