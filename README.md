# Cancer Diagnostic Clustering Analysis

This project aims to perform clustering analysis on a cancer diagnostic dataset using K-means and Agglomerative Clustering techniques. The goal is to segment the dataset into meaningful clusters based on feature similarities and evaluate the performance of the clustering models.

## Key Features:

1. **Imported File**: Loaded the cancer diagnostic dataset for analysis.
2. **Checked for Null Values**: Identified and handled missing values in the dataset.
3. **K-Means Clustering**: Applied K-means clustering to segment the data into clusters.
4. **WCSS Score**: Calculated the Within-Cluster Sum of Squares (WCSS) to evaluate clustering performance.
5. **Optimal Clustering**: Tested different numbers of clusters and selected the best one based on the WCSS score.
6. **Cluster Centers**: Printed the cluster centers for the optimal number of clusters.
7. **Cluster Labels**: Created a new column for the predicted cluster labels.
8. **Hierarchical Clustering**: Used Scipy to plot hierarchical clustering.
9. **Agglomerative Clustering**: Applied Agglomerative Clustering to further analyze the dataset.
10. **Label Creation**: Created a label column for the predicted clusters from Agglomerative Clustering.
11. **Label Counts**: Displayed the counts for each cluster label.
12. **Silhouette Score**: Achieved a silhouette score of 68%, indicating the quality of the clustering.

## Why This Project?

This project demonstrates how clustering techniques can be used to segment data in the medical field. By applying both K-means and Agglomerative Clustering, we explore different ways of clustering the cancer diagnostic data to gain insights into underlying patterns. The project also emphasizes the importance of evaluating clustering performance using metrics like WCSS and silhouette score.

## What You Will Learn:

- How to apply K-means and Agglomerative Clustering to real-world datasets.
- The process of selecting the optimal number of clusters using WCSS.
- Evaluating clustering performance using silhouette scores.
- Practical insights into handling missing values and preprocessing data for clustering.

## Conclusion

Through the application of clustering techniques, this project successfully segmented the cancer diagnostic dataset into meaningful clusters. K-means and Agglomerative Clustering provided valuable insights into the data structure, with a silhouette score of 68% indicating reasonable clustering quality. This project highlights the importance of clustering techniques in medical data analysis and provides a comprehensive approach to handling clustering tasks in machine learning.
