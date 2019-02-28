import numpy as np 
import pickle
def compute_distance(feature_centers,datapoint):
    '''
        make changes here!!
    
    '''


    # print(feature_centers.shape,' center ',datapoint.shape,' datapoint')
    return np.sum(np.power(datapoint-feature_centers,2),axis=1)

def kmeans(data,num_cluster_centers,epochs=1000):
    cluster = np.zeros((data.shape[0],1))
    center_indexes = np.random.random_integers(0,data.shape[0],num_cluster_centers)
    feature_centers = data[center_indexes]
    # print('feature centers: ',feature_centers.shape)    
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
    num_cluster_centers = feature_centers.shape[0]
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
    
    return feature_centers

def process_dataset(file_path,store=False,output_file_path=None):
    with open(file_path,'r') as f:
        textual_data = []
        protein = ""
        string = ""
        max_len = -1
        for line in f.readlines():
            # print(line)
            if ">" in line:
                continue
            else:
                protein += line[:-1]
                if line[-2] == "*":
                    protein = protein[:-1]
                    textual_data += [protein]
                    max_len = max(max_len,len(protein))
                    protein = ""
                elif line[-1] == "*":
                    protein = protein
                    textual_data += [protein]
                    max_len = max(max_len,len(protein))
                    protein = ""
                string += protein

        string_list = sorted(list(set(string)))# extract the amino acids
        mapping_from_acid_to_vector = {acid : string_list.index(acid) for acid in string_list}
        dataset = np.zeros((len(textual_data),max_len,len(mapping_from_acid_to_vector)))
        for data_string in range(len(textual_data)):
            for acid_position in range(len(textual_data[data_string])):
                dataset[data_string][acid_position][mapping_from_acid_to_vector[textual_data[data_string][acid_position]]] = 1
        if store:
            with open('textual_'+output_file_path,'wb') as f:
                pickle.dump(textual_data,f)
            with open('matrix_'+output_file_path,'wb') as f:
                pickle.dump(dataset,f)
        return textual_data,mapping_from_acid_to_vector,dataset


if __name__ == "__main__":
    # data = process_dataset('dataset.txt',True,'processed_data.txt')
    textual_data,mapping_from_acid_to_vector,data_matrix = process_dataset('dataset.txt',store=True,output_file_path='data.txt')
