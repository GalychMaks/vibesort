import abc
import logging
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from vibesort.utils import get_audio_file_paths, save_playlists


class BaseVibeSorter(abc.ABC):
    """
    Abstract base class for audio clustering into playlists based on extracted features.
    """

    def __init__(self):
        """
        Initializes the vibe sorter and configures the logger.
        """
        self.logger = self._configure_logger()

    def _configure_logger(self) -> logging.Logger:
        """
        Configures a logger for the vibe sorter instance.

        :return: Configured logger instance.
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def run(self, folder_path: str, num_clusters: int, output_file: str = None) -> dict:
        """
        Executes the vibe sorting process:
            - extracts features
            - clusters them
            - saves the resulting playlists

        :param folder_path: Path to the folder containing audio files.
        :param num_clusters: Number of clusters to group the audio files into.
        :param output_file: Optional path to save the result as a JSON file.
        :return: Dictionary containing the clustered playlists.
        """
        file_paths = get_audio_file_paths(folder_path)
        self.logger.info(f"Found {len(file_paths)} audio files to process.")

        self.logger.info("Extracting features from audio files...")
        features = []
        for path in tqdm(file_paths):
            try:
                feature = self.extract_features_for_file(path)
                features.append(feature)
            except Exception as e:
                self.logger.warning(f"Failed to extract features from {path}: {e}")

        if not features:
            self.logger.error("No features extracted. Exiting.")
            return None

        self.logger.debug(f"Feature array shape: {np.array(features).shape}")

        self.logger.info(f"Running clusterization with {num_clusters} clusters")
        labels = self._cluster_features(features, num_clusters)

        if output_file:
            self.logger.info(f"Saving playlists to: '{output_file}'")

        result = save_playlists(labels, file_paths, output_file)
        self.logger.info("All done. 🎧")

        return result

    def _cluster_features(self, features: List[np.ndarray], num_clusters: int) -> List[int]:
        """
        Clusters the extracted feature vectors using KMeans.

        :param features: List of feature vectors extracted from audio files.
        :param num_clusters: Number of clusters to form.
        :return: List of cluster labels for each feature vector.
        """
        features_array = np.array(features)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        return kmeans.fit_predict(features_array)

    @abc.abstractmethod
    def extract_features_for_file(self, path: str) -> np.ndarray:
        """
        Extracts a feature vector from a given audio file.

        :param path: Path to the audio file.
        :return: Feature vector representing the audio file.
        """
        pass
