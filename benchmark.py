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
        "--mspca", help="Transform using MultispatiPCA.", action="store_true"
    )
    parser.add_argument(
        "--neighbors", help="Neighbors, 'delaunay' or int.", required=False, default=4
    )
    parser.add_argument(
        "--n_rings",
        help="number of rings for grid. (only used with stereoseq)",
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        "--n_pcs", help="Number of components.", required=False, type=int
    )
    parser.add_argument(
        "--n_genes",
        help="Number of genes. (only used with stereoseq)",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--svg", help="Use SVG instead of HVG (only stereo-seq).", action="store_true"
    )
    parser.add_argument("--seed", help="Random seed.", required=True, type=int)

    args = parser.parse_args()

    import numpy as np

    from utils import process_imagingbased, process_stereoseq

    np.random.seed(args.seed)

    neighbors = args.neighbors if args.neighbors == "delaunay" else int(args.neighbors)

    if args.stereoseq:
        label_df = process_stereoseq(
            path=args.h5ad,
            spatial_weight=args.spatial_weight,
            SVG=args.svg,
            seed=args.seed,
            n_pcs=args.n_pcs,
            n_genes=args.n_genes,
            n_neighs=neighbors,
            n_rings=args.n_rings,
            mspca=args.mspca,
        )

    else:
        label_df = process_imagingbased(
            path=args.h5ad,
            seed=args.seed,
            msPCA=args.mspca,
            neighbors=neighbors,
            spatial_weight=args.spatial_weight,
            n_pcs=args.n_pcs,
        )

    label_df.to_csv(args.out, sep="\t", index_label="")


if __name__ == "__main__":
    main()
