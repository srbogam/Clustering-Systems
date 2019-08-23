# Clustering Systems

This repository contains implementations of **K-Means Clustering** and
**Hierarchical Clustering**.

### K-Means

### Hierarchical Clustering

The current hierarchical clustering algorithms uses **agglomeration** to create the
clusters and uses the following linkages:
1. Single Link (MIN)
2. Complete Link (MAX)
3. Group Average (AVG)

The agglomerative hierarchical clustering script is divided into two classes:
1. Agglomerative_Hierarchical
2. Proximity_Matrix

These are further divided as:
1. ***matrix_min()***: Returns the current minimum value in the passed matrix.
2. ***min_cluster_distance()***: Returns the minimum distance between clusters.
3. ***max_cluster_distance()***: Returns the maximum distance between clusters.
4. ***avg_cluster_distance()***: Returns the average distance between clusters.
5. ***matrix_gen()***: Generates a new proximity matrix after cluster formation.
6. ***clustering()***: Clusters points agglomeratively and returns the linkage matrix.
7. ***distance()***: Calculates distance between points.
8. ***raw_matrix()***: Generates the proximity matrix for the first time from data.

### Results

The algorithms were run on a dataset consisting of amino acid sequences. The results are
published as dendrograms:

**K-Means Clustering**


**Hierarchical clustering**:

1. ***Single Link***

![Single Link](https://github.com/nsurampu/Clustering-Systems/blob/master/Hierarchical%20Clustering/Agglomerative/agglomerative-single-link-dendrogram.png)

2. ***Complete Link***

![Complete Link](https://github.com/nsurampu/Clustering-Systems/blob/master/Hierarchical%20Clustering/Agglomerative/agglomerative-complete-link-dendrogram.png)

3. ***Group Average***

![Group Average](https://github.com/nsurampu/Clustering-Systems/blob/master/Hierarchical%20Clustering/Agglomerative/agglomerative-single-link-dendrogram.png)

### Libraries Used
1. Numpy
2. Scipy
3. Matplotlib


