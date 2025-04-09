import librosa
import numpy as np

from vibesort.base_sorter import BaseVibeSorter


class AlgoVibeSorter(BaseVibeSorter):
    def extract_features_for_file(self, path: str):
        y, sr = librosa.load(path, sr=None, mono=True)

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossings = librosa.feature.zero_crossing_rate(y)

        # All output must be 1D arrays before concatenation
        features = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(chroma, axis=1),
                np.array([np.mean(spectral_centroid)]),
                np.array([np.mean(spectral_bandwidth)]),
                np.array([np.mean(spectral_rolloff)]),
                np.array([np.mean(zero_crossings)]),
            ]
        ).astype(np.float32)

        return features
