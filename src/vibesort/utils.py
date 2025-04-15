import json
import os
from typing import List


def get_audio_file_paths(folder_path: str) -> List[str]:
    """
    Retrieves the paths of audio files within a specified folder.

    :param folder_path: Path to the folder containing audio files.
    :return: List of full file paths to supported audio files.
    """
    supported_ext = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported_ext)
    ]


def save_playlists(labels, file_paths, output_file: str = None):
    """
    Groups audio into playlists based on corresponding labels and optionally writes to a JSON file.

    :param labels: Sequence of labels corresponding to each audio file.
    :param file_paths: Sequence of file paths to be grouped into playlists.
    :param output_file: Optional path to the output JSON file.
    :return: Dictionary representing the generated playlists.
    """
    playlists = {}
    for label, path in zip(labels, file_paths):
        label = int(label)
        playlists.setdefault(label, []).append(os.path.basename(path))

    output = {"playlists": [{"id": int(pid), "songs": songs} for pid, songs in playlists.items()]}

    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

    return output
