import argparse

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
        "-o", "--output", default="playlists_v1.json", help="Output file name"
    )
    args = parser.parse_args()

    sorter = AlgoVibeSorter()
    sorter.run(
        folder_path=args.path, num_clusters=args.num_clusters, output_file=args.output
    )


if __name__ == "__main__":
    main()
