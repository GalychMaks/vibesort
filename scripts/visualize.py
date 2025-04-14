import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimensions(features: np.ndarray, method: str = "pca") -> np.ndarray:
    """
    Reduce the dimensionality of feature data to 2D using PCA or t-SNE.

    :param features: The high-dimensional feature array.
    :param method: The dimensionality reduction method to use ('pca' or 'tsne').
    :returns: The 2D reduced feature array.
    """
    if method == "tsne":
        n_samples = len(features)
        perplexity = min(30, max(5, n_samples // 3))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        reducer = PCA(n_components=2)
    return reducer.fit_transform(features)


def plot_comparison(
    reduced_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    method: str,
    output_path: str,
) -> None:
    """
    Plot a side-by-side comparison of 2D reduced features from different sorters.

    :param reduced_dict: A dictionary mapping sorter names to their reduced feature arrays.
    :param labels_dict: A dictionary mapping sorter names to their cluster labels.
    :param method: The dimensionality reduction method used ('pca' or 'tsne').
    :param output_path: The path where the plot image will be saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colormap = plt.get_cmap("tab10")

    for ax, (sorter_name, reduced) in zip(axes, reduced_dict.items()):
        labels = labels_dict[sorter_name]
        unique_labels = np.unique(labels)

        for i, label in enumerate(unique_labels):
            points = reduced[labels == label]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                color=colormap(i),
                label=f"Cluster {label}",
                edgecolors="w",
                linewidths=0.5,
            )

        ax.set_title(f"{sorter_name.capitalize()} ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """
    Main execution function to extract features, perform clustering, reduce dimensions,
    and plot visual comparisons using both PCA and t-SNE.
    """
    from vibesort.algo_sorter import AlgoVibeSorter
    from vibesort.deep_sorter import DeepVibeSorter
    from vibesort.utils import get_audio_file_paths

    sorters: Dict[str, object] = {
        "algo": AlgoVibeSorter(),
        "deep": DeepVibeSorter(),
    }
    folder_path = "data/audio"
    num_clusters = 3
    file_paths: List[str] = get_audio_file_paths(folder_path)

    features_dict: Dict[str, np.ndarray] = {}
    labels_dict: Dict[str, np.ndarray] = {}

    # Extract features and labels
    for sorter_name, sorter in sorters.items():
        features = [sorter.extract_features_for_file(p) for p in file_paths]
        labels = sorter._cluster_features(features, num_clusters)
        features_dict[sorter_name] = np.array(features)
        labels_dict[sorter_name] = np.array(labels)

    # Plot comparisons
    for method in ["pca", "tsne"]:
        reduced_dict = {
            name: reduce_dimensions(feats, method) for name, feats in features_dict.items()
        }
        output_path = f"output/comparison_{method}.png"
        plot_comparison(reduced_dict, labels_dict, method, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
