from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp


def get_anndata(path):
    X = sp.io.mmread(path / "counts.mtx").tocsr()

    observations = pd.read_table(path / "observations.tsv", index_col=0)
    features = pd.read_table(path / "features.tsv", index_col=0)

    coordinates = (
        pd.read_table(path / "coordinates.tsv", index_col=0)
        .loc[observations.index, :]
        .to_numpy()
    )

    adata = ad.AnnData(
        X=X, obs=observations, var=features, obsm={"spatial": coordinates}
    )

    return adata


def preprocess_anndata(adata, seed=42, genes=1000, n_pcs=30, min_cells=10):
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # sc.pp.scale(adata, zero_center=False)


def process_imagingbased(
    path: Path,
    seed: int,
    msPCA: bool = True,
    neighbors="delaunay",
    spatial_weight: float = 0.8,
    n_pcs: int = 30,
):
    import squidpy as sq
    from multispaeti import MultispatiPCA
    from spatialleiden import distance2connectivity, search_resolution

    np.random.seed(seed=seed)
    adata = ad.read_h5ad(path)

    n_clusters = adata.obs["ground_truth"].nunique()

    sc.pp.log1p(adata)
    # sc.pp.scale(adata, zero_center=False)

    if neighbors == "delaunay":
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
        directed = False
    elif isinstance(neighbors, int):
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=neighbors)
        directed = True
    else:
        raise Exception

    adata.obsp["spatial_connectivities"] = distance2connectivity(
        adata.obsp["spatial_distances"]
    )

    if msPCA:
        rep = "X_mspca"
        adata.obsm[rep] = MultispatiPCA(
            n_pcs, connectivity=adata.obsp["spatial_connectivities"]
        ).fit_transform(adata.X)

    else:
        sc.pp.pca(adata, n_comps=n_pcs, random_state=seed)
        rep = "X_pca"

    sc.pp.neighbors(adata, use_rep=rep, random_state=seed)

    _ = search_resolution(
        adata,
        n_clusters,
        latent_kwargs={"random_state": seed},
        spatial_kwargs={
            "layer_ratio": spatial_weight,
            "directed": (False, directed),
            "seed": seed,
        },
    )

    df = adata.obs[["spatialleiden"]].copy()
    df.columns = ["label"]

    return df


def process_stereoseq(
    path: Path,
    seed: int,
    SVG: bool,
    spatial_weight: float = 0.8,
    n_pcs: int = 30,
    n_genes: int = 3_000,
    n_neighs: int = 4,
    n_rings: int = 1,
    mspca: bool = True,
):
    import squidpy as sq
    from multispaeti import MultispatiPCA
    from spatialleiden import search_resolution

    np.random.seed(seed=seed)
    adata = ad.read_h5ad(path)
    preprocess_anndata(adata, genes=n_genes, n_pcs=n_pcs, seed=seed)

    n_clusters = adata.obs["ground_truth"].nunique()

    sq.gr.spatial_neighbors(
        adata, coord_type="grid", n_neighs=n_neighs, n_rings=n_rings
    )

    if SVG:
        sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
        genes = adata.uns["moranI"].nlargest(n_genes, columns="I", keep="all").index
    else:
        genes = adata.var_names[adata.var["highly_variable"]]

    if mspca:
        rep = "X_mspca"
        adata.obsm[rep] = MultispatiPCA(
            n_pcs, connectivity=adata.obsp["spatial_connectivities"]
        ).fit_transform(adata[:, genes].X.toarray())
    else:
        sc.pp.pca(
            adata,
            n_comps=n_pcs,
            mask_var=adata.var_names.isin(genes),
            random_state=seed,
        )
        rep = "X_pca"

    sc.pp.neighbors(adata, use_rep=rep, random_state=seed)

    _ = search_resolution(
        adata,
        n_clusters,
        latent_kwargs={"random_state": seed},
        spatial_kwargs={
            "layer_ratio": spatial_weight,
            "directed": (False, False),
            "seed": seed,
        },
    )

    df = adata.obs[["spatialleiden"]].copy()
    df.columns = ["label"]

    return df