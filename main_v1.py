import argparse
import json

from vibesort import AlgoVibeSorter


def main():
    parser = argparse.ArgumentParser(
        description="Cluster songs using classical audio features."
    )
    parser.add_argument(
        "-p", "--path", default="data/audio/", help="Path to folder with audio files"
    )
    parser.add_argument(
        "-n",
        "--num_clusters",
        type=int,
        default=3,
        help="Number of playlists to generate",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Output file name (optional)"
    )
    args = parser.parse_args()

    sorter = AlgoVibeSorter()
    playlists = sorter.run(
        folder_path=args.path, num_clusters=args.num_clusters, output_file=args.output
    )
    print(json.dumps(playlists, indent=2))


if __name__ == "__main__":
    main()
