import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimensions(features: np.ndarray, method: str = "pca") -> np.ndarray:
    """
    Reduce feature dimensions to 2D using PCA or t-SNE.
    """
    if method == "tsne":
        n_samples = len(features)
        perplexity = min(30, max(5, n_samples // 3))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        reducer = PCA(n_components=2)
    return reducer.fit_transform(features)


def plot_clusters(reduced: np.ndarray, labels: np.ndarray, output_path: str) -> None:
    """
    Plot reduced features with cluster labels and save to file.
    """
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colormap = plt.get_cmap("tab10", len(unique_labels))

    for i, label in enumerate(unique_labels):
        cluster_points = reduced[labels == label]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=colormap(i),
            label=f"Cluster {label}",
            edgecolors="w",
            linewidths=0.5,
        )

    plt.title("Audio Feature Clusters (2D)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Clusters")
    plt.grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def visualize_clusters(
    features: List[np.ndarray],
    labels: List[int],
    output_image_path: str,
    method: str = "pca",
) -> None:
    """
    Visualize clustered feature vectors in 2D and save the plot to an image file.
    """
    features_array = np.array(features)
    labels_array = np.array(labels)

    reduced = reduce_dimensions(features_array, method)
    plot_clusters(reduced, labels_array, output_image_path)


def main():
    from vibesort.algo_sorter import AlgoVibeSorter
    from vibesort.deep_sorter import DeepVibeSorter
    from vibesort.utils import get_audio_file_paths

    sorters = {"deep": DeepVibeSorter(), "algo": AlgoVibeSorter()}
    methods = ["pca", "tsne"]
    folder_path = "data/audio"
    num_clusters = 3

    file_paths = get_audio_file_paths(folder_path)

    for sorter_name, sorter_instance in sorters.items():
        features = [sorter_instance.extract_features_for_file(p) for p in file_paths]
        labels = sorter_instance._cluster_features(features, num_clusters)

        for method in methods:
            output_path = f"output/{sorter_name}_{method}.png"
            visualize_clusters(
                features, labels, output_image_path=output_path, method=method
            )
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
