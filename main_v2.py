import argparse

from vibesort import DeepVibeSorter


def main():
    parser = argparse.ArgumentParser(
        description="Cluster songs using MERT audio embeddings."
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
        "--output", default="playlists_v2.json", help="Output file name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MERT-v1-95M",
        help="MERT model to use",
    )

    args = parser.parse_args()

    sorter = DeepVibeSorter(mert_model_name=args.model)
    sorter.run(
        folder_path=args.path, num_clusters=args.num_clusters, output_file=args.output
    )


if __name__ == "__main__":
    main()
