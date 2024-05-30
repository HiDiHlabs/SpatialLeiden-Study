from pathlib import Path

import anndata as ad
import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import squidpy as sq
from multispaeti import MultispatiPCA
from scipy.sparse import find


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


def search_resolution(
    adata, ncluster, start=1, step=0.1, n_iterations=15, seed=42, **kwargs
):
    # adapted from SpaGCN.search_res (https://github.com/jianhuupenn/SpaGCN)
    res = start
    sc.tl.leiden(adata, resolution=res, random_state=seed, **kwargs)
    old_ncluster = adata.obs["leiden"].cat.categories.size
    iter = 1
    while old_ncluster != ncluster:
        old_sign = 1 if (old_ncluster < ncluster) else -1
        sc.tl.leiden(
            adata, resolution=res + step * old_sign, random_state=seed, **kwargs
        )
        new_ncluster = adata.obs["leiden"].cat.categories.size
        if new_ncluster == ncluster:
            res = res + step * old_sign
            # print(f"Recommended res = {res:.2f}")
            return res
        new_sign = 1 if (new_ncluster < ncluster) else -1
        if new_sign == old_sign:
            res = res + step * old_sign
            # print(f"Res changed to {res:.2f}")
            old_ncluster = new_ncluster
        else:
            step = step / 2
            # print(f"Step changed to {step:.2f}")
        if iter > n_iterations:
            # print("Exact resolution not found")
            # print(f"Recommended res =  {res:.2f}")
            return res
        iter += 1
    # print(f"Recommended res = {res:.2f}")
    return res


def preprocess_adata(adata, seed=42, genes=1000, n_pcs=30):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(adata, random_state=seed)


def run_leiden(adata, n_clusters, seed, **kwargs):
    res = search_resolution(adata, n_clusters, seed=seed, **kwargs)

    df = adata.obs[["leiden"]]
    df.columns = ["label"]
    return df, res


def _build_igraph(adjacency, directed=True):
    # adapted from scanpy
    sources, targets, weights = find(adjacency)
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])
    g.add_edges(list(zip(sources, targets)))
    g.es["weight"] = weights
    if g.vcount() != adjacency.shape[0]:
        raise RuntimeError(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g


def search_spatial_resolution_multiplex(
    adata, ncluster, start=0.4, step=0.1, n_iterations=15, seed=42, **kwargs
):
    # adapted from SpaGCN.search_res (https://github.com/jianhuupenn/SpaGCN)
    res = start
    spatial_partition_kwargs = kwargs.pop("spatial_partition_kwargs", dict())
    spatial_partition_kwargs["resolution_parameter"] = res
    leiden_multiplex(
        adata, spatial_partition_kwargs=spatial_partition_kwargs, seed=seed, **kwargs
    )
    old_ncluster = adata.obs["leiden_multiplex"].cat.categories.size
    iter = 1
    while old_ncluster != ncluster:
        old_sign = 1 if (old_ncluster < ncluster) else -1
        spatial_partition_kwargs["resolution_parameter"] = res + step * old_sign
        leiden_multiplex(
            adata,
            spatial_partition_kwargs=spatial_partition_kwargs,
            seed=seed,
            **kwargs,
        )
        new_ncluster = adata.obs["leiden_multiplex"].cat.categories.size
        if new_ncluster == ncluster:
            res = res + step * old_sign
            # print(f"Recommended res = {res:.2f}")
            return res
        new_sign = 1 if (new_ncluster < ncluster) else -1
        if new_sign == old_sign:
            res = res + step * old_sign
            # print(f"Res changed to {res:.2f}")
            old_ncluster = new_ncluster
        else:
            step = step / 2
            # print(f"Step changed to {step:.2f}")
        if iter > n_iterations:
            # print("Exact resolution not found")
            # print(f"Recommended res = {res:.2f}")
            return res
        iter += 1
    # print(f"Recommended res = {res:.2f}")
    return res


def leiden_multiplex(
    adata,
    key_added: str = "leiden_multiplex",
    directed: tuple[bool, bool] = (True, True),
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type=la.RBConfigurationVertexPartition,
    scale_graph_weights: tuple[bool, bool] = (False, False),
    layer_weights: tuple[int, int] = (1, 1),
    spatial_partition_kwargs=None,
    latent_partition_kwargs=None,
    diff_threshold: float = 1e-05,
    seed=42,
):
    spatial_distance_key = "spatial_connectivities"
    latent_distance_key = "connectivities"

    latent_distances = adata.obsp[latent_distance_key]
    spatial_distances = adata.obsp[spatial_distance_key]

    if scale_graph_weights[0]:
        percentile = np.percentile(latent_distances.data, 95)
        latent_distances = latent_distances.multiply(1 / percentile)
    if scale_graph_weights[1]:
        percentile = np.percentile(spatial_distances.data, 95)
        spatial_distances = spatial_distances.multiply(1 / percentile)

    adjacency_latent = _build_igraph(latent_distances, directed=directed[0])
    adjacency_spatial = _build_igraph(spatial_distances, directed=directed[1])

    # parameterise the partitions
    if spatial_partition_kwargs is None:
        spatial_partition_kwargs = dict()
    if latent_partition_kwargs is None:
        latent_partition_kwargs = dict()

    if use_weights:
        spatial_partition_kwargs["weights"] = "weight"
        latent_partition_kwargs["weights"] = "weight"

    latent_part = partition_type(adjacency_latent, **latent_partition_kwargs)
    spatial_part = partition_type(adjacency_spatial, **spatial_partition_kwargs)
    optimiser = la.Optimiser()
    optimiser.set_rng_seed(seed)

    diff = optimiser.optimise_partition_multiplex(
        [latent_part, spatial_part],
        layer_weights=list(layer_weights),
        n_iterations=n_iterations,
    )

    adata.obs[key_added] = np.array(latent_part.membership)
    adata.obs[key_added] = adata.obs[key_added].astype("category")


def run_leiden_multiplex(adata, n_clusters, seed, **kwargs):
    res = search_spatial_resolution_multiplex(adata, n_clusters, seed=seed, **kwargs)

    df = adata.obs[["leiden_multiplex"]]
    df.columns = ["label"]
    return df, res


def distance2connectivity(distances):
    connectivity = distances.copy()
    connectivity.data = 1 - (connectivity.data / connectivity.data.max())
    return connectivity


def process_imagingbased(
    path: Path,
    seed: int,
    log: bool = True,
    msPCA: bool = True,
    neighbors="delaunay",
    spatial_weight: float = 0.8,
    n_pcs: int = 30,
):
    np.random.seed(seed=seed)
    adata = ad.read_h5ad(path)

    n_clusters = adata.obs["ground_truth"].nunique()

    if log:
        sc.pp.log1p(adata)

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
        adata.obsm["X_mspca"] = MultispatiPCA(
            n_pcs, connectivity=adata.obsp["spatial_connectivities"]
        ).fit_transform(adata.X)

    else:
        rep = "X"

    sc.pp.neighbors(adata, use_rep=rep, random_state=seed)

    label_leiden, res = run_leiden(adata, n_clusters, seed=seed)

    label_leiden_multi, _ = run_leiden_multiplex(
        adata,
        n_clusters,
        directed=(False, directed),
        scale_graph_weights=(False, False),
        layer_weights=(1, spatial_weight),
        latent_partition_kwargs={"resolution_parameter": res},
        seed=seed,
    )
    return label_leiden_multi


def process_stereoseq(
    path: Path,
    seed: int,
    SVG: bool,
    spatial_weight: float = 0.8,
    n_pcs: int = 30,
    n_genes: int = 3_000,
):
    np.random.seed(seed=seed)
    adata = ad.read_h5ad(path)
    preprocess_adata(adata, genes=n_genes, n_pcs=n_pcs, seed=seed)

    n_clusters = adata.obs["ground_truth"].nunique()

    sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=4)

    if SVG:
        sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
        genes = (
            adata.uns["moranI"]
            .nlargest(n_genes, columns="I", keep="all")
            .index.to_list()
        )
    else:
        genes = adata.var_names[adata.var["highly_variable"]].to_list()

    adata.obsm["X_mspca"] = MultispatiPCA(
        n_pcs, connectivity=adata.obsp["spatial_connectivities"]
    ).fit_transform(adata[:, genes].X.toarray())

    sc.pp.neighbors(adata, use_rep="X_mspca", random_state=seed)

    label_leiden, res = run_leiden(adata, n_clusters, seed=seed)

    label_leiden_multi, _ = run_leiden_multiplex(
        adata,
        n_clusters,
        directed=(False, False),
        scale_graph_weights=(False, False),
        layer_weights=(1, spatial_weight),
        latent_partition_kwargs={"resolution_parameter": res},
        seed=seed,
    )
    return label_leiden_multi
