# WeatherIndex
Development of a Weather Severity Index to Improve AIS-Based Ship Arrival Time Predictions in Inland Waterway Transport

# Development of a Weather Severity Index to Improve AIS-Based Ship Arrival Time Predictions in Inland Waterway Transport
This repository contains the code used in the research paper titled **"Development of a Weather Severity Index to Improve AIS-Based Ship Arrival Time Predictions in Inland Waterway Transport"**. The study aims to enhance the accuracy of ship arrival time predictions by incorporating weather severity indices derived from meteorological data.

## Table of Contents
- [Introduction](#introduction)
- [Data Source](#data-source)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Introduction
This project investigates the effectiveness of various clustering algorithms in analyzing meteorological data to improve Automatic Identification System (AIS)-based ship arrival time predictions in inland waterway transport (IWT). The weather patterns were grouped using different clustering methods, and their performance was evaluated using the Silhouette Score. The findings of this research are detailed in the associated research paper.

## Data Source
The meteorological data used in this research was obtained from the Koninklijk Nederlands Meteorologisch Instituut (KNMI). The data can be accessed via the following link:
[KNMI Climate Data](https://www.knmi.nl/nederland-nu/klimatologie/daggegevens)

## Repository Structure
- README.md # This README file
- clustering_analysis.py # Main Python script for data processing and clustering analysis
- Data_KNMI.zip # Folder containing the raw KNMI weather data .txt files
- KNMI_data_cleaned.zip # Cleaned weather_data.csv
- weather_clusters.csv # Output file with clustering results

## Installation
To run the code in this repository, you need to have Python v3 installed along with the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- umap-learn

## Usage
Prepare the Data:
1. Place all raw KNMI weather data files in the Data_KNMI folder.
2. Run the Clustering Analysis:
3. Execute the clustering_analysis.py script to process the data and perform clustering analysis.
4. View Results: The clustering results and weather index rankings are saved in the weather_clusters.csv file. Visualizations of the clustering results using PCA, t-SNE, and UMAP are displayed as plots.

## Results
The code evaluates the performance of different clustering algorithms (KMeans, DBSCAN, Agglomerative Clustering with Ward, Complete, and Average linkages, and Gaussian Mixture Models) and ranks the weather severity based on the clustering results. The findings, including the optimal clustering method and weather severity rankings, are detailed in the research paper.

## Contributing
Contributions to this project are welcome. If you have suggestions or improvements, feel free to create a pull request or open an issue.
