#!/usr/bin/env python


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i", "--input", help="Path to input file.", required=True, type=Path
    )
    parser.add_argument(
        "-o", "--out", help="Path of outfile.", required=True, type=Path
    )
    parser.add_argument("--reg_exp", help="# factors", type=int, required=True)
    parser.add_argument("--k", help="# factors", type=int, default=20)
    parser.add_argument("--seed", help="Random seed.", default=0, type=int)
    
    args = parser.parse_args()

    import torch
    from popari import tl
    from popari.model import SpiceMix
    from popari.io import save_anndata

    n_preiter = 5
    n_iter = 200

    torch_context = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": torch.float32,
    }

    model = SpiceMix(
        K=args.k,
        lambda_Sigma_x_inv=10**-args.reg_exp,
        dataset_path=args.input,
        random_state=args.seed,
        initial_context=torch_context,
        torch_context=torch_context,
    )

    # Run
    model.train(num_preiterations=n_preiter, num_iterations=n_iter)

    tl.preprocess_embeddings(model, normalized_key="normalized_X")
    save_anndata(args.out, [model.datasets[0]])


if __name__ == "__main__":
    main()