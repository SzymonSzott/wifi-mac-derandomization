from sklearn.cluster import DBSCAN
import numpy as np

def dbscan_clustering(distance_matrix, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on the given distance matrix.

    Parameters:
    - distance_matrix: A 2D list or numpy array representing pairwise distances.
    - eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: Minimum number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - labels: Cluster labels for each point in the dataset.
    """

    # Convert distance matrix to a numpy array if it's not already
    distance_matrix = np.array(distance_matrix)

    # Create a DBSCAN clustering model
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')

    # Fit the model and predict labels
    labels = dbscan_model.fit_predict(distance_matrix)

    # Return the model and cluster labels
    return dbscan_model, labels

def extract_clusters(dbscan_model, macs):
    """
    Extract clusters from the fitted DBSCAN model.

    Parameters:
    - dbscan_model: Fitted DBSCAN model.
    - macs: List of MAC addresses corresponding to the points.

    Returns:
    - clusters: A dictionary mapping cluster labels to lists of MAC addresses.
    """

    clusters = {}
    for idx, label in enumerate(dbscan_model.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(macs[idx])
    
    return clusters