import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def normalize_coordinates(coordinates):
    min_vals = np.min(coordinates, axis=0)
    max_vals = np.max(coordinates, axis=0)
    return (coordinates - min_vals) / (max_vals - min_vals), min_vals, max_vals

def denormalize_coordinates(normalized_coords, min_vals, max_vals):
    return normalized_coords * (max_vals - min_vals) + min_vals

def visualize_clusters(data, labels, centroids, output_path="static/cluster_plot.png"):
    """
    Visualize clusters and centroids.

    Parameters:
        data (numpy.ndarray): The normalized data points.
        labels (numpy.ndarray): Cluster labels for the data points.
        centroids (numpy.ndarray): Coordinates of the cluster centroids.
        output_path (str): File path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    # Scatter plot for data points
    for label in np.unique(labels):
        cluster_points = data[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
    # Scatter plot for centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', label='Centroids')
    plt.legend()
    plt.grid(True)
    plt.title("Cluster Visualization")
    plt.xlabel("Longitude (Normalized)")
    plt.ylabel("Latitude (Normalized)")

    # Save plot instead of showing it
    plt.savefig(output_path)
    plt.close()

def visualize_routing(grouped_clusters, unvisitable, output_path="static/routing_plot.png"):
    """
    Visualize the routing for all clusters, showing schedules and unvisited locations.

    Parameters:
        grouped_clusters (dict): Clustered and scheduled locations with reasons.
        unvisitable (list): List of unvisited locations.
        output_path (str): File path to save the routing visualization.
    """
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10.colors  # Colormap
    cluster_colors = {}

    # Plot each cluster
    for cluster_id, cluster_data in grouped_clusters.items():
        schedule = cluster_data["schedule"]
        color = colors[cluster_id % len(colors)]  # Assign a unique color to each cluster
        cluster_colors[cluster_id] = color

        # Extract coordinates and plot them
        coordinates = [loc["coordinates"] for loc in schedule]
        names = [f"{i + 1}: {loc['name']}" for i, loc in enumerate(schedule)]

        # Plot points
        for i, coord in enumerate(coordinates):
            plt.scatter(coord[1], coord[0], color=color, label=f"Cluster {cluster_id}" if i == 0 else "", s=100)
            plt.text(coord[1], coord[0], names[i], fontsize=9, ha="right")

        # Plot connections
        for i in range(len(coordinates) - 1):
            start = coordinates[i]
            end = coordinates[i + 1]
            plt.plot([start[1], end[1]], [start[0], end[0]], color=color, linestyle="--", alpha=0.7)

    # Plot unvisited locations
    for location in unvisitable:
        coord = location.coordinates
        plt.scatter(coord[1], coord[0], color="red", label="Unvisitable", marker="x", s=100)
        plt.text(coord[1], coord[0], location.name, fontsize=9, ha="right", color="red")

    # Add legend and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper left")
    plt.title("Routing Visualization")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Save the plot
    plt.savefig(output_path)
    plt.close()
    
def generate_schedule_table(schedule_table, cluster_id, output_path="static/schedule_table_cluster_{cluster_id}.png"):
    """
    Generate a table visualization for a daily schedule.

    Parameters:
        schedule_table (list): List of dictionaries containing schedule details.
        cluster_id (int): The cluster ID for the schedule.
        output_path (str): Path to save the schedule image.

    Returns:
        str: Path to the saved table image.
    """
    # Convert the schedule to a pandas DataFrame
    df = pd.DataFrame(schedule_table)

    # Create a figure and axis to draw the table
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 1))  # Adjust height dynamically
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save the table as an image
    output_path = output_path.format(cluster_id=cluster_id)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    return output_path