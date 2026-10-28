import hdbscan
import numpy as np


def hdbscan_clustering(
    distance_matrix,
    min_cluster_size=5,
    min_samples=None,
    cluster_selection_epsilon=0.0,
):
    """
    Perform HDBSCAN clustering on the given distance matrix.

    Parameters:
    - distance_matrix: A 2D list or numpy array representing pairwise distances.
    - min_cluster_size: Minimum number of samples in a cluster.
    - min_samples: Number of samples in a neighborhood for a point to be considered as a core point.
                   If None, defaults to min_cluster_size.
    - cluster_selection_epsilon: Distance threshold for merging nearby clusters.

    Returns:
    - hdbscan_model: Fitted HDBSCAN model.
    - labels: Cluster labels for each point in the dataset.
    """

    distance_matrix = np.array(distance_matrix)

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        cluster_selection_epsilon=cluster_selection_epsilon,
    )

    labels = hdbscan_model.fit_predict(distance_matrix)

    return hdbscan_model, labels


def extract_clusters(hdbscan_model, macs):
    """
    Extract clusters from the fitted HDBSCAN model.
    """

    clusters = {}
    for idx, label in enumerate(hdbscan_model.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(macs[idx])

    return clusters