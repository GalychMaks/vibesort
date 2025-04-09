# Vibesort

Tool for clustering songs into playlists based on their audio characteristics and vibes. Vibesort analyzes audio features to automatically group similar songs together, creating cohesive playlists that match different moods or styles.

## Installation

Requires Python 3.8 or higher.

```bash
# Clone the repository
git clone https://github.com/yourusername/vibesort.git
cd vibesort

# Install the package
pip install -e .
```

## Usage

You can use Vibesort from the command line:

```bash
python main_v2.py -p /path/to/audio/folder -n 3 --output playlists.json
```

Arguments:

- `-p, --path`: Path to folder containing audio files (default: "data/audio/")
- `-n, --num_clusters`: Number of playlists to generate (default: 3)
- `--output`: Output file name for the generated playlists (default: "playlists_v2.json")
