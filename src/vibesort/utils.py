import json
import os
from typing import List


def get_audio_file_paths(folder_path: str) -> List[str]:
    supported_ext = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported_ext)
    ]


def save_playlists(labels, file_paths, output_file: str):
    playlists = {}
    for label, path in zip(labels, file_paths):
        label = int(label)
        playlists.setdefault(label, []).append(os.path.basename(path))

    output = {
        "playlists": [
            {"id": int(pid), "songs": songs} for pid, songs in playlists.items()
        ]
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
