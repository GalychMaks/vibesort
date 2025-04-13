import argparse

from vibesort import AlgoVibeSorter, DeepVibeSorter


def main():
    parser = argparse.ArgumentParser(
        description="Cluster songs using classical audio features or MERT embeddings."
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
        "-o",
        "--output",
        default=None,
        help="Output file name (auto-selected if not provided)",
    )
    parser.add_argument(
        "--method",
        choices=["algo", "deep"],
        default="deep",
        help="Clustering method: 'algo' for classical features, "
        "'deep' for deep learning embeddings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MERT-v1-95M",
        help="MERT model to use (only used with 'deep' method)",
    )

    args = parser.parse_args()

    if args.method == "algo":
        output_file = args.output or "playlists_v1.json"
        sorter = AlgoVibeSorter()
    else:
        output_file = args.output or "playlists_v2.json"
        sorter = DeepVibeSorter(mert_model_name=args.model)

    sorter.run(
        folder_path=args.path,
        num_clusters=args.num_clusters,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
