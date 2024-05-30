#!/usr/bin/env python

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i", "--h5ad", help="Path to h5ad file.", required=True, type=Path
    )
    parser.add_argument(
        "-o", "--out", help="Path of outputfile (tsv).", required=True, type=Path
    )
    parser.add_argument(
        "--stereoseq",
        help="Whether to process StereoSeq or image-based data.",
        action="store_true",
    )
    parser.add_argument(
        "--spatial_weight", help="Weight for spatial layer.", required=True, type=float
    )
    parser.add_argument(
        "--log", help="Log transform counts (only image-based).", action="store_true"
    )
    parser.add_argument(
        "--mspca",
        help="Transform using MultispatiPCA (only image-based).",
        action="store_true",
    )
    parser.add_argument(
        "--neighbors",
        help="Neighbor method. (only used with image-based)",
        required=False,
    )
    parser.add_argument(
        "--n_pcs",
        help="Neighbor method. (only used with --mspca)",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--n_genes",
        help="Neighbor method. (only used with stereoseq)",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--svg",
        help="Random seed. (only used with stereoseq)",
        action="store_true",
    )
    parser.add_argument("--seed", help="Random seed.", required=True, type=int)

    args = parser.parse_args()

    import numpy as np

    from leiden_utils import process_imagingbased, process_stereoseq

    np.random.seed(args.seed)

    if args.stereoseq:
        label_df = process_stereoseq(
            path=args.h5ad,
            spatial_weight=args.spatial_weight,
            SVG=args.svg,
            seed=args.seed,
            n_pcs=args.n_pcs,
            n_genes=args.n_genes,
        )

    else:
        neighbors = (
            args.neighbors if args.neighbors == "delaunay" else int(args.neighbors)
        )
        label_df = process_imagingbased(
            path=args.h5ad,
            seed=args.seed,
            log=args.log,
            msPCA=args.mspca,
            neighbors=neighbors,
            spatial_weight=args.spatial_weight,
            n_pcs=args.n_pcs,
        )

    label_df.to_csv(args.out, sep="\t", index_label="")


if __name__ == "__main__":
    main()