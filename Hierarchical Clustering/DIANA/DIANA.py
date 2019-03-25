import os
import pickle
import math
import numpy
import copy
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from pprint import pprint

class Divisive_Hierarchical:
    """
    This class contains functions which comprise the core implementation of divisive
	hierarchical clustering. The method is implemented is commonly referred to as DIANA.

    @author : Naren Surampudi
    """

    def total_distance(self, point, cluster, matrix):
        """
        This function calculates the summation of all distances bewtween a point and all
        points in the passed cluster.

        Parameters
        ----------
        point : int
            The index of a point in the data.
        cluster : list
            A cluster of points.
        matrix : list
            The proximity matrix of the data.

        Returns
        -------
        type : int
            The summation of distances between a point and all points in the cluster.
        """
        cdist = 0
        for cpoint in cluster:
            cdist = cdist + matrix[point][cpoint]

        return cdist

    def avg_cluster_distance(self, matrix, cluster1, cluster2):
        """
		This function calculates the average distance between two clusters of points.
	    Parameters
	    ----------
	    matrix : list
	        The matrix to be operated upon.
	    cluster1 : list
	        Cluster of points.
	    cluster2 : list
	        Cluster of points.
	    Returns
	    -------
	    type : int
	        Average distance between the two passed clusters.
        """
        dist_list = []

        if isinstance(cluster1, int):
        	cluster1 = [cluster1]
        elif isinstance(cluster2, int):
        	cluster2 = [cluster2]
        for point1 in cluster1:
        	for point2 in cluster2:
        		dist = matrix[point1][point2]
        		if dist > 0:
        			dist_list.append(dist)

        return sum(dist_list) / ((len(cluster1) * len(cluster2)))

    def clustering(self, n, clusters, matrix):
        """
        This function performs divisive clustering. This makes use of the DIANA algorithm.

        Parameters
        ----------
        n : int
            Total number of data points.
        clusters : list
            List of all the present clusters.
        matrix : list
            Proximity matrix of the data points.

        Returns
        -------
        type : numpy array
            Linkage matrix to be used for plotting the clustering dendrogram.
        """
        K = len(clusters)
        linkage_matrix = numpy.zeros(shape=(len(matrix[0])-1, 4))
        counter = 0
        split_clusters = []
        cluster_ids = []

        while K < n:
            temp_K = K
            temp_clusters = []
            for cluster in clusters:
                if cluster not in cluster_ids:
                    cluster_ids.append(cluster)
                cluster_A = copy.deepcopy(cluster)
                # print(cluster_A)
                if len(cluster_A) > 1:
                    cluster_B = []
                    flag = True
                    while flag:
                        avg_dist = 0
                        mv_point = None
                        if len(cluster_B) == 0:
                            for point in cluster_A:
                                temp_avg_dist = self.total_distance(point, cluster_A, matrix) / len(cluster_A)
                                if temp_avg_dist > avg_dist:
                                    avg_dist = temp_avg_dist
                                    mv_point = point
                            if mv_point == None:
                                temp_clusters.append(cluster_A)
                                flag = False
                            else:
                                # print(mv_point, avg_dist)
                                cluster_B.append(mv_point)
                                cluster_A.remove(mv_point)
                        else:
                            while flag:
                                avg_dist = 0
                                mv_point = None
                                for point in cluster_A:
                                    temp_avg_dist_A = self.total_distance(point, cluster_A, matrix) / len(cluster_A)
                                    temp_avg_dist_B = self.total_distance(point, cluster_B, matrix) / len(cluster_B)
                                    temp_avg_dist = temp_avg_dist_A - temp_avg_dist_B
                                    if temp_avg_dist > avg_dist:
                                        avg_dist = temp_avg_dist
                                        mv_point = point

                                if avg_dist == 0:
                                    flag = False
                                    # print(len(cluster_A), len(cluster_B))
                                    break
                                if mv_point != None:
                                    # print(mv_point)
                                    cluster_B.append(mv_point)
                                    cluster_A.remove(mv_point)

                    if len(cluster_B) != 0:
                        temp_clusters.append(cluster_A)
                        temp_clusters.append(cluster_B)
                        split_clusters.append(cluster)
                        cluster_ids.append(cluster_A)
                        cluster_ids.append(cluster_B)
                        avg_cdist = self.avg_cluster_distance(matrix, cluster_A, cluster_B)
                        cluster_qt = len(cluster_A) + len(cluster_B)
                        # cluster_ids[tuple(cluster)]-1
                        # cluster_ids[tuple(cluster_B)]-1
                        linkage_matrix[n-len(clusters)-1] = [cluster_ids.index(cluster_A), cluster_ids.index(cluster_B), avg_cdist, n-cluster_qt+1]
                        # print(cluster_A)
                        # print(cluster)
                else:
                    temp_clusters.append(cluster)

            clusters = []
            for cluster in temp_clusters:
                clusters.append(cluster)

            K = len(clusters)
            if temp_K == K:
                break
        # print(clusters)
        linkage_matrix = numpy.flip(linkage_matrix[1:], 0)

        return linkage_matrix

class Proximity_Matrix:
    """
	This class is called in the very beginning, when calculating the proximity matrix for
	the first time from the data.
	"""

    def distance(self, sample1, sample2):
        """Short summary.
        Parameters
        ----------
        sample1 : list
        A sample in the data.
        sample2 : list
        A sample in the data.
        Returns
        -------
        type : float
        The distance between the two samples.
        """
        edist = 0
        for i in range(len(sample1)):
        	dist = 0
        	for j in range(len(sample1[i])):
        		dist = dist + abs(sample1[i][j] - sample2[i][j])
        		edist = edist + dist

        return math.sqrt(edist)

    def raw_matrix(self, data):
        """
        This function calculates the first proximity matrix.
        Parameters
        ----------
        data : list
            The processed data, obtained from raw data.
        Returns
        -------
        type : list
            The very first proximity matrix.
        """
        matrix = []

        for sample1 in data:
            l = []
            for sample2 in data:
                dist = self.distance(sample1, sample2)
            l.append(dist)
            matrix.append(l)

        return matrix

if __name__ == "__main__":

    divisive = Divisive_Hierarchical()
    proximity = Proximity_Matrix()

    matrix_file = "proximity_matrix.pkl"
    if os.path.isfile(matrix_file):
        matrix_f = open(matrix_file, 'rb')
        matrix = pickle.load(matrix_f)
        matrix_f.close()
    else:
        dfile = open('matrix_data.txt', 'rb')
        data = pickle.load(dfile)
        # data = data[0:100]
        matrix = proximity.raw_matrix(data)
        matrix_f = open(matrix_file, 'wb')
        pickle.dump(matrix, matrix_f)
        matrix_f.close()
        dfile.close()

    points = [p for p in range(0, len(matrix))]
    initial_cluster = [points]
    linkage_matrix = divisive.clustering(len(matrix), initial_cluster, matrix)
    print(linkage_matrix)

    fig = plt.figure(figsize=(8, 4))
    dendrogram = dendrogram(linkage_matrix)   # Draw dendrogram of final clusters.
    plt.show()
    # some = [[1], [2]]
    # print(some.index([2]))
