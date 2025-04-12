import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from vibesort.base_sorter import BaseVibeSorter


class DeepVibeSorter(BaseVibeSorter):
    SUPPORTED_MODELS = {
        "MERT-v1-330M",
        "MERT-v1-95M",
        "MERT-v0-public",
        "MERT-v0",
        "music2vec-v1",
    }

    def __init__(self, mert_model_name: str = "m-a-p/MERT-v1-95M"):
        super().__init__()

        if mert_model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported MERT model: '{mert_model_name}'.\n"
                f"Supported models are: {', '.join(sorted(self.SUPPORTED_MODELS))}"
            )

        mert_model_name = f"m-a-p/{mert_model_name}"

        self.logger.info(f"Loading MERT model: {mert_model_name}")
        self.model = AutoModel.from_pretrained(mert_model_name, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            mert_model_name, trust_remote_code=True
        )

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Target sample rate expected by MERT
        self.target_sr = self.processor.sampling_rate

        # Lazy initialization of resampler
        self.resampler = None

    def extract_features_for_file(self, path: str):
        # Load audio file
        audio, sr = sf.read(path)

        # Convert stereo to mono
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        audio_tensor = torch.from_numpy(audio).float()
        if sr != self.target_sr:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            audio_tensor = self.resampler(audio_tensor)

        # Tokenize input
        inputs = self.processor(
            audio_tensor.numpy(), sampling_rate=self.target_sr, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get hidden states from MERT
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            self.logger.debug(f"Outputs shape: {outputs.hidden_states[0].shape}")

        # Stack hidden states: [13 layers, time steps, 768]
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        self.logger.debug(
            f"All layer hidden states shape: {all_layer_hidden_states.shape}"
        )

        # Average across time -> [13, 768]
        time_avg = all_layer_hidden_states.mean(dim=1)
        self.logger.debug(f"Time average shape: {time_avg.shape}")

        # Then average across layers -> [768]
        embedding = time_avg.mean(dim=0)
        self.logger.debug(f"Embedding shape: {embedding.shape}")

        # Return as NumPy array
        return embedding.cpu().numpy().astype(np.float32)
