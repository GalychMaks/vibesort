import numpy as np
import openl3
import soundfile as sf

from vibesort.base_sorter import BaseVibeSorter


class DeepVibeSorter(BaseVibeSorter):
    def __init__(self):
        super().__init__()

        # Load the OpenL3 model once
        self.model = openl3.models.load_audio_embedding_model(
            input_repr="mel256", content_type="music", embedding_size=512
        )

    def extract_features_for_file(self, path: str):
        # Load audio file
        audio, sr = sf.read(path)

        # If stereo, convert to mono
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)

        # Extract embedding (batch of embeddings over time)
        emb, _ = openl3.get_audio_embedding(
            audio, sr, model=self.model, hop_size=1.0, center=True, verbose=False
        )

        # Pool over time (mean pooling)
        features = np.mean(emb, axis=0).astype(np.float32)
        return features
