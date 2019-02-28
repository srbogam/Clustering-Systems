import numpy as np 

def compute_distance(feature_centers,datapoint):
    # print(feature_centers.shape,' center ',datapoint.shape,' datapoint')
    return np.sum(np.power(datapoint-feature_centers,2),axis=1)

def kmeans(data,num_cluster_centers,epochs=1000):
    cluster = np.zeros((data.shape[0],1))
    center_indexes = np.random.random_integers(0,data.shape[0],num_cluster_centers)
    feature_centers = data[center_indexes]
    # print('feat cent: ',feature_centers.shape)    
    # print(center_indexes)   
    for epoch in range(epochs):
        distances = np.zeros((num_cluster_centers,1))
        for datapoint in range(data.shape[0]):
            # print('datapoint ',datapoint)
            distances = compute_distance(feature_centers,data[datapoint,:])
            cluster_index = np.argmin(distances)
            cluster[datapoint,0] = cluster_index
        for i in range(num_cluster_centers):
            cluster_points_indices = np.argwhere(cluster == i)
            cluster_points = data[cluster_points_indices[:,0]]
            # print('points: ',cluster_points.shape,cluster_points_indices[:,0])
            if cluster_points.shape[0] != 0:
                # print('centers: ',feature_centers.shape)
                feature_centers[i] = np.mean(cluster_points,axis=0)
                # print('centers: ',feature_centers.shape)
        # if epoch % 10 == 0:        
        #    print('epoch: ',epoch)
    num_cluster_centers = num_neurons = feature_centers.shape[0]
    cluster = np.zeros((data.shape[0],1))
    l2_norm = np.zeros((data.shape[0],1))
    beta = np.zeros((num_cluster_centers,1))
    
    for datapoint in range(data.shape[0]):
        distances = np.zeros((num_cluster_centers,1))
        # print('datapoint ',datapoint)
        distances = compute_distance(feature_centers,data[datapoint,:])
        cluster_index = np.argmin(distances)
        cluster[datapoint,0] = cluster_index
        l2_norm[datapoint,0] = distances[cluster_index]

    for i in range(num_cluster_centers):
        cluster_points_indices = np.argwhere(cluster == i)
        distances_l2 = np.power(l2_norm[cluster_points_indices[:,0]],0.5)
        beta[i] = 1/2*np.power(np.sum(distances_l2)/distances_l2.shape[0],2)
    
    return feature_centers,beta
