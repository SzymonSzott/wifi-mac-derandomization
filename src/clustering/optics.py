from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt
import os

def optics_clustering(
    distance_matrix,
    min_samples=2,
    xi=0.05,
    min_cluster_size=15,
    max_eps=np.inf,
    ):
    """
    Perform OPTICS clustering on the given distance matrix.

    Parameters:
    - distance_matrix: A 2D list or numpy array representing pairwise distances.
    - min_samples: Minimum number of samples in a neighborhood for a point to be considered as a core point.
    - eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.

    Returns:
    - labels: Cluster labels for each point in the dataset.
    """

    # Convert distance matrix to a numpy array if it's not already
    distance_matrix = np.array(distance_matrix)

    # Create an OPTICS clustering model
    optics_model = OPTICS(
    min_samples=min_samples,
    xi=xi,
    min_cluster_size=min_cluster_size,
    max_eps=max_eps,
    metric="precomputed",
    )

    # Fit the model to the distance matrix
    optics_model.fit(distance_matrix)

    # Return the cluster labels
    return optics_model, optics_model.labels_

def plot_optics_reachability(optics_model, labels, output_path="plots/reachability_plot.svg"):
    """
    Plot the reachability plot for the OPTICS model.

    Parameters:
    - optics_model: Fitted OPTICS model.
    """

    ordering = optics_model.ordering_
    reachability = optics_model.reachability_[ordering]
    labels_ordered = optics_model.labels_[ordering]

    plt.figure(figsize=(12, 6))
    for klass in np.unique(labels_ordered):
        mask = labels_ordered == klass
        plt.plot(np.arange(len(reachability))[mask], reachability[mask],
                marker='.', linestyle='', label=f"Cluster {klass}" if klass != -1 else "Noise")

    plt.title("Reachability Plot (by ordering)")
    plt.ylabel("Reachability Distance")
    plt.xlabel("Ordered Point Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def extract_clusters(optics_model, macs):
    """
    Extract clusters from the fitted OPTICS model.

    Parameters:
    - optics_model: Fitted OPTICS model.

    Returns:
    - clusters: A dictionary mapping cluster labels to lists of point indices.
    """

    clusters = {}
    for idx, label in enumerate(optics_model.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(macs[idx])
    
    return clusters