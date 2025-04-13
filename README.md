# Vibesort

Tool for clustering songs into playlists based on their audio characteristics and vibes. Vibesort analyzes audio features to automatically group similar songs together, creating cohesive playlists that match different moods or styles.

## Installation

### Virtual Environment (Recommended)

It's best practice to use a virtual environment to keep your project dependencies isolated:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# [Optional] install pytorch according to instructions
# https://pytorch.org/get-started/
```

Once activated, you can proceed with either inference-only or development setup.

### Inference only

If you only need the package for inference, install it directly from GitHub:

```bash
pip install git+https://github.com/GalychMaks/vibesort.git
```

### Development

If you want to contribute or make changes to the code:

```bash
# Clone the repository
git clone https://github.com/GalychMaks/vibesort.git
cd vibesort

# Install the package with development dependencies
pip install -e .[dev]

# Set up pre-commit hooks
pre-commit install
```

## Usage

### Command Line Interface (CLI)

After installation, you can use Vibesort directly from the command line:

```bash
vibesort -p /path/to/audio/folder -n 3 --method deep
```

Arguments:

- `-p, --path`: Path to folder containing audio files (default: "data/audio/")
- `-n, --num_clusters`: Number of playlists to generate (default: 3)
- `-o, --output`: Output file name for the generated playlists (optional, if not provided playlists will only be returned)
- `--method`: Clustering method to use: 'deep' (default) or 'algo'
- `--model`: MERT model to use with the deep method (default: "MERT-v1-95M")

### Python API

You can also use Vibesort programmatically in your Python code:

```python
from vibesort import DeepVibeSorter, AlgoVibeSorter

# Using deep learning-based clustering (recommended)
sorter = DeepVibeSorter(mert_model_name="MERT-v1-95M")
playlists = sorter.run(
    folder_path="path/to/audio/folder",
    num_clusters=3
)

# Or using classical audio feature-based clustering
sorter = AlgoVibeSorter()
playlists = sorter.run(
    folder_path="path/to/audio/folder",
    num_clusters=3
)
```

The output will be a Python dictionary containing the generated playlists, where each song is grouped into a cluster based on its audio characteristics. If an output file is provided, the playlists will also be saved to that file as JSON.

## References

This project is built upon several powerful open-source technologies:

### Deep Learning

- [MERT (Music Embedding through Representation Transformers)](https://huggingface.co/m-a-p/MERT-v1-95M) - Pre-trained music understanding model used for extracting deep audio features
- [🤗 Transformers](https://github.com/huggingface/transformers) - Framework for using state-of-the-art transformer models

### Audio Processing

- [librosa](https://librosa.org/) - Python package for music and audio analysis, used for classical feature extraction
- [soundfile](https://github.com/bastibe/python-soundfile) - Library for reading and writing audio files
- [torchaudio](https://pytorch.org/audio) - Audio processing tools from the PyTorch ecosystem

### Machine Learning

- [scikit-learn](https://scikit-learn.org/) - Used for K-means clustering of audio features
- [PyTorch](https://pytorch.org/) - Deep learning framework used with MERT models

### Development Tools

- [Ruff](https://github.com/astral-sh/ruff) - Fast Python linter and formatter
