import librosa
import numpy as np
from librosa.feature import (
    chroma_stft,
    mfcc,
    rms,
    spectral_bandwidth,
    spectral_centroid,
    spectral_rolloff,
    zero_crossing_rate,
)

from vibesort.base_sorter import BaseVibeSorter


class AlgoVibeSorter(BaseVibeSorter):
    """
    An implementation of BaseVibeSorter that extracts audio features using librosa.
    """

    def extract_features_for_file(self, path: str) -> np.ndarray:
        """
        Extracts a feature vector from an audio file using signal processing techniques.

        Features extracted include:
        - MFCC (mean and standard deviation)
        - Chroma frequencies
        - Spectral centroid
        - Spectral bandwidth
        - Spectral rolloff
        - Zero crossing rate
        - Root mean square (RMS) energy
        - Tempo

        :param path: Path to the audio file.
        :return: 1D numpy array representing the extracted feature vector.
        """
        y, sr = librosa.load(path, sr=None, mono=True)

        # Extract features
        mfcc_ft = mfcc(y=y, sr=sr, n_mfcc=13)  # shape: (13, T)
        chroma_ft = chroma_stft(y=y, sr=sr)  # shape: (12, T)
        spectral_centroid_ft = spectral_centroid(y=y, sr=sr)  # shape: (1, T)
        spectral_bandwidth_ft = spectral_bandwidth(y=y, sr=sr)  # shape: (1, T)
        spectral_rolloff_ft = spectral_rolloff(y=y, sr=sr)  # shape: (1, T)
        zero_crossings_rate_ft = zero_crossing_rate(y)  # shape: (1, T)
        rms_ft = rms(y=y)  # shape: (1, T)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # scalar

        # Concatenate features into a single vector
        features = np.concatenate(
            [
                np.mean(mfcc_ft, axis=1),  # shape: (13,)
                np.std(mfcc_ft, axis=1),  # shape: (13,)
                np.mean(chroma_ft, axis=1),  # shape: (12,)
                np.array([np.mean(spectral_centroid_ft)]),  # shape: (1,)
                np.array([np.mean(spectral_bandwidth_ft)]),  # shape: (1,)
                np.array([np.mean(spectral_rolloff_ft)]),  # shape: (1,)
                np.array([np.mean(zero_crossings_rate_ft)]),  # shape: (1,)
                np.array([np.mean(rms_ft)]),  # shape: (1,)
                np.array(tempo),  # shape: (1,)
            ]
        ).astype(np.float32)  # Final shape: (44,)

        return features
