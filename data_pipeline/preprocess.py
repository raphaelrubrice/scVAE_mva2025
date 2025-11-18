
import scanpy as sc


def preprocess_pbmc3k(raw_path, output_h5ad="data/pbmc3k_preprocessed.h5ad"):
    """
    Full Scanpy preprocessing pipeline for PBMC3k (as used in SCVAE paper).
    Produces a filtered, normalized, log-transformed, HVG-subsetted AnnData.
    """
    adata = sc.read_10x_mtx(raw_path, var_names="gene_ids")
    adata.var_names_make_unique()

    # filter out lowly represented cells/genes
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # annotate mitochondrial genes (not sure why but done in the scanpy documentaiton tutorials)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # Filter by QC thresholds
    adata = adata[
        (adata.obs.n_genes_by_counts < 2500)
        & (adata.obs.n_genes_by_counts > 200)
        & (adata.obs.pct_counts_mt < 5)
    ].copy()
    adata.layers["counts"] = adata.X.copy()

    # Normalize plus log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Highly variable genes (Seurat flavor)
    sc.pp.highly_variable_genes(
        adata,
        layer="counts",
        n_top_genes=2000,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        flavor="seurat_v3",
    )
    adata = adata[:, adata.var.highly_variable].copy()

    # Regress out unwanted sources of variation, scale
    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(adata, max_value=10)

    adata.write_h5ad(output_h5ad)
    print(f"Saved preprocessed data to {output_h5ad}")

    return adata