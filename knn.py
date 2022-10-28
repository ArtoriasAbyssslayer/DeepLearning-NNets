from collections import Counter
import math 
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
def knn(k,query_sample,data,distance_function,problem_fn):
    # List that stores distances and the  
    neigbor_dist_indices_buff = []
    
    
    for indx, sample in enumerate(data):        
        # Calculate the distance matrix
        distance = distance_function(sample[:-1],query_sample)
        # Append the Neighbor-Distance Buffer
        neigbor_dist_indices_buff.append(distance,indx)
        
        
    # Sort the buffer sorted(iterable,key,reverse) python provided fu
    sorted_neigbor_buff = sorted(neigbor_dist_indices_buff)
    
    k_neirest_neigbors = sorted_neigbor_buff[:k]
    # Get the labels of K-First entries
    
    k_nearest_labels = [data[i][-1] for distance,i in k_neirest_neigbors]
    
    # Return the nearest neighbors sorted and the predicted label based on the mode or mean 
    return k_neirest_neigbors, problem_fn(k_nearest_labels)






""" Euclidean Distance it implements L2 norm between 2 points """
def euclidean_distance(pointA,pointB):
    squared_distance_sum  = 0;
    for i in range(len(pointA)):
        squared_distance_sum += (pointA[i]-pointB[i])**2
        
    return math.sqrt(squared_distance_sum)

""" Regression = find most common real value ~ mean val, Classification = find the most common label ~ mode """
 
""" mean is the problem_fun for regression task """
def mean(labels):
    return sum(labels) / len(labels)



""" mode is the problem_func for classification task """
def mode(labels):
    return Counter(labels).most_common(1)[0][0]        