from fastapi import APIRouter, HTTPException
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from fastapi import BackgroundTasks
from app.models import ClusteringInput
from app.clustering import tensorflow_kmeans
from app.scheduling import parallel_schedule_clusters, handle_unvisitable
# from app.utils import visualize_clusters, visualize_routing, generate_schedule_table
from app.evaluation import (
    compute_silhouette_score,
    # compute_davies_bouldin_index,
    # compute_intra_cluster_distance,
)
from concurrent.futures import ThreadPoolExecutor
import sys

sys.dont_write_bytecode = True

# Initialize the API router for clustering-related endpoints
clustering_router = APIRouter()

@clustering_router.post("/cluster/", summary="Cluster Locations and Generate Schedules")
def cluster_data(data: ClusteringInput): 
    """
    Endpoint to perform clustering on location data and generate schedules
    (Intinya gabungin semua fungsi yang udh dibuat di modul-modul lain utk generate response)
    - Takes in a set of location points and performs clustering based on the specified number of clusters
    - Schedules the visits within daily time constraints, handles any unvisitable locations, and returns the clustering results

    Arguments:
        data (ClusteringInput): Input data containing location points, number of clusters, and daily start/end times
        Info lebih lanjut soal tipe datanya dkk cek modul models
        
    Returns a dict containing grouped clusters with schedules and unvisitable locations (response)
    """
    start_time = time.time()
    
    # Extract coordinates, names, bla bla bla and create a mapping of location names to their details
    coordinates = np.array([loc.coordinates for loc in data.points])
    names = [loc.name for loc in data.points]
    locations = {loc.name: loc for loc in data.points}
    num_clusters = data.num_clusters
    daily_start = data.daily_start_time
    daily_end = data.daily_end_time

    # Input validation
    if num_clusters < 1:
        raise HTTPException(status_code=400, detail="Number of clusters must be at least 1")

    if len(coordinates) < num_clusters:
        raise HTTPException(status_code=400, detail="Number of clusters cannot exceed number of points")

    # Normalize coordinates to scale features between 0 and 1
    # Knp minmax? Tak tahu distribusinya normal atau agak laen
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(coordinates)

    if num_clusters == 1:
        # Berhubung cuma 1 cluster, lempar semua ke dalam, gosah cluster lagi
        best_labels = np.zeros(len(coordinates), dtype=int)  # All points belong to cluster 0
        best_centroids = normalized_data.mean(axis=0).reshape(1, -1)  # Centroid = mean of all points
        best_clusters = {0: list(locations.values())}  # All locations in cluster 0

        # Tak perlu best_metrics since we don't perform any clustering
        # best_metrics = {
        #     "silhouette_score": None,
        #     "davies_bouldin_index": None,
        #     "intra_cluster_distance": None,
        # }
    else:
        # Perform parallel clustering to find the best cluster
        best_clusters, best_labels, best_centroids, best_metrics = parallel_find_best_clusters(
            normalized_data, locations, num_clusters, num_iterations=8
        )

    # Visualize the clustering results
    # cluster_plot_path = "static/cluster_plot.png"
    # visualize_clusters(
    #     data=normalized_data,
    #     labels=best_labels,
    #     centroids=best_centroids,
    #     output_path=cluster_plot_path,
    # )

    # Schedule the clustered locations within daily time constraints
    grouped_clusters = parallel_schedule_clusters(best_clusters, daily_start, daily_end)

    # Extract locations that couldn't be visited within the constraints
    unvisitable_locations = [
        loc for cluster in grouped_clusters.values() for loc in cluster["unvisitable"]
    ]

    # Attempt to reschedule unvisitable locations (last chance)
    adjusted_result = handle_unvisitable(unvisitable_locations, grouped_clusters, daily_start, daily_end)
    grouped_clusters = adjusted_result["clusters"]
    final_unvisitable = adjusted_result["unvisitable"]

    # Generate schedule tables for each cluster
    # cluster_tables = {}
    # for cluster_id, cluster_data in grouped_clusters.items():
    #     schedule_table = [
    #         {
    #             "Name": loc["name"],
    #             "Start Time": loc["start_time"],
    #             "End Time": loc["end_time"],
    #             "Opening Hours": f"{locations[loc['name']].opening_hours} - {locations[loc['name']].closing_hours}",
    #             "Duration (hours)": locations[loc["name"]].duration,
    #             "Reason": loc.get("reason", "N/A"),
    #             "Proximity": loc.get("proximity_to_next", "N/A"),
    #         }
    #         for loc in cluster_data["schedule"]
    #     ]
    #     table_path = generate_schedule_table(schedule_table, cluster_id)
    #     cluster_tables[cluster_id] = table_path

    # Visualize routing based on the scheduled clusters
    # routing_plot_path = "static/routing_plot.png"
    # visualize_routing(
    #     grouped_clusters=grouped_clusters,
    #     unvisitable=final_unvisitable,
    #     output_path=routing_plot_path
    # )
    
    end_time = time.time()
    
    total_runtime = end_time - start_time

    # Compile final response
    response = {
        "grouped_clusters": [
            {
                "cluster": cluster_id,
                "schedule": [
                    {
                        "name": loc["name"],
                        "start_time": loc["start_time"],
                        "end_time": loc["end_time"],
                    }
                    for loc in cluster_data["schedule"]
                ],
                # "unvisitable": cluster_data["unvisitable"],  
            }
            for cluster_id, cluster_data in grouped_clusters.items()
        ],
        "final_unvisitable": [ 
            {"name": loc.name, "reason": "Time constraints prevent scheduling"}
            for loc in final_unvisitable
        ],
        "total_runtime_seconds": total_runtime,
        # "metrics": best_metrics, 
        # "visualization": {  
        #     "cluster_plot_path": cluster_plot_path,
        #     "routing_plot_path": routing_plot_path
        # },
    }

    return response

# Do we need this multithreading too? Or is it too much? (For clustering)
def parallel_find_best_clusters(normalized_data, locations, num_clusters, num_iterations):
    """
    Perform K-means clustering in parallel and prioritize balanced clusters.

    - Executes multiple K-means clustering iterations concurrently using parallel processing.
    - Evaluates each clustering based on a composite score that considers the silhouette
    score and cluster balance. The best clustering = highest composite score.

    Parameters:
        normalized_data (numpy.ndarray): Normalized dataset with rows representing data points.
        locations (dict): Mapping of location names to their details.
        num_clusters (int): The number of clusters to form.
        num_iterations (int): Number of clustering iterations to perform in parallel.

    Returns:
        tuple:
            - best_clusters (dict): Cluster assignments with lists of location details.
            - best_labels (numpy.ndarray): Cluster labels assigned to each data point.
            - best_centroids (numpy.ndarray): Centroid positions of the best clustering configuration.
            - best_metrics (dict): Evaluation metrics (silhouette score and cluster balance score) for the best clustering configuration.
    """
    best_clusters = None
    best_labels = None
    best_centroids = None
    best_metrics = None
    best_composite_score = float("-inf")

    def cluster_and_evaluate(_):
        """
        Perform a single clustering iteration and compute evaluation metrics.

        This helper function executes the K-means clustering and calculates:
        - Silhouette score: Measures cluster cohesion and separation.
        - Cluster balance score: Measures variance in cluster sizes.

        Returns:
            tuple: Centroids, labels, silhouette score, and cluster balance score.
        """
        centroids, labels = tensorflow_kmeans(normalized_data, num_clusters)
        silhouette = compute_silhouette_score(normalized_data, labels)
        cluster_balance_score = compute_cluster_balance_score(labels, num_clusters)
        return centroids, labels, silhouette, cluster_balance_score

    # Run clustering iterations in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cluster_and_evaluate, range(num_iterations)))

    # Select the best cluster configuration based on composite score
    for centroids, labels, silhouette, cluster_balance in results:
        composite_score = (0.7 * silhouette) + (0.3 * (1 - cluster_balance))  # Combine metrics
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_labels = labels
            best_centroids = centroids
            best_metrics = {
                "silhouette_score": silhouette,
                "cluster_balance_score": cluster_balance,
            }
            best_clusters = {i: [] for i in range(num_clusters)}
            for name, cluster_id in zip(locations.keys(), labels):
                best_clusters[int(cluster_id)].append(locations[name])

    return best_clusters, best_labels, best_centroids, best_metrics

def compute_cluster_balance_score(labels, num_clusters):
    """
    Compute a score for cluster balance based on the variance of cluster sizes.
    Explanation: 
        - Balanced clustering has clusters with similar sizes = lower variance.
        - Balance score normalized by the mean cluster size for comparability.

    Returns float of normalized variance of cluster sizes (lower values indicate better balance)
    """
    cluster_counts = [np.sum(labels == i) for i in range(num_clusters)]
    balance_score = np.var(cluster_counts) / np.mean(cluster_counts)  # Normalize by mean
    return balance_score