from collections import Counter
import numpy
import torch
'''
    knn function
    Description : Implements K-Nearest-Neighbors algorithm from scratch
    Args:
    - k: number of nearest neighbors selected
    - query_sample: Data Sample that has the k-nearest-sample
    - data: Set of preprocessed data for the task
    - distance_function: Objective fuction with which near concept is define here is used L2 norm - Euclidean Distance
    - Problem_fn: Knn can be used both for classification and regression problem but in a different way this function is 
                  responsible to define the use case.
'''
def knn_custom(k,query_sample,data,problem_fn):
    # List that stores distances and the  
    neighbor_dist_buff = []
    neighbor_indeces = []
    
    for indx, sample in enumerate(data):        
        # Calculate the distance matrix
        distance = compute_distances(sample[:-1],query_sample)
        
        # Append the Neighbor-Distance Buffer
        neighbor_dist_buff.append(distance)
        
    # Sort the buffer sorted(iterable,key,reverse) python provided fu
    sorted_neighbor_buff = numpy.sort(neighbor_dist_buff)
    sorted_neighbor_buff = sorted_neighbor_buff[::-1]
    k_nearest_neigbors = sorted_neighbor_buff[:k]
    
    # Get the labels of K-First entries
    for p in k_nearest_neigbors:
        r = numpy.argwhere(neighbor_dist_buff==p)
        k_nearest_labels = data[r]
    # Return the nearest neighbors sorted and the predicted label based on the mode or mean 
    return k_nearest_neigbors, problem_fn(k_nearest_labels)






""" Euclidean Distance it implements L2 norm between 2 points """
def compute_distances(X,query):
    size_X = X.shape[0]
    size_query = query.shape[0]
    dists = numpy.zeros((size_X,size_query))
    X = X.numpy()
    X = X[0,:,:]
    query = query.numpy()
    query = query[0,:,:]
    temp=numpy.subtract(X,query)
    # for i in range(size_X):
    #    for j in range(size_query):
    #       dists[i][j] = euclidean_distance(X[i],query[i])
    dists = numpy.sqrt(numpy.dot(temp,temp))
    return dists
def euclidean_distance(pointA,pointB):
    squared_distance_sum  = 0;
    for i in range(len(pointA)):
        squared_distance_sum += (pointA[i]-pointB[i])**2
        
    return numpy.sqrt(squared_distance_sum)

""" Regression = find most common real value ~ mean val, Classification = find the most common label ~ mode """
 
""" mean is the problem_fun for regression task """
def mean(labels):
    return sum(labels) / len(labels)



""" mode is the problem_func for classification task """
def mode(labels):
    return Counter(labels).most_common(1)[0][0]        
