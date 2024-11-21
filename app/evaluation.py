import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from collections import Counter

def compute_silhouette_score(data, labels):
    return silhouette_score(data, labels)

def compute_davies_bouldin_index(data, labels):
    return davies_bouldin_score(data, labels)

def compute_intra_cluster_distance(data, labels, centroids):
    total_distance = 0
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        total_distance += np.sum(cdist(cluster_points, [centroids[i]]))
    return total_distance

def check_cluster_balance(labels, max_locations_per_day=5):
    cluster_counts = Counter(labels)
    overloaded_clusters = [i for i, count in cluster_counts.items() if count > max_locations_per_day]
    return cluster_counts, overloaded_clusters