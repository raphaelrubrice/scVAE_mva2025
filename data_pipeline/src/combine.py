# src/combine.py
"""
Combine individual PBMC AnnData objects into unified artifacts
--------------------------------------------------------------
Loads per‑dataset .mtx folders, annotates them via load_anndata(),
freezes category vocabularies, optionally harmonizes gene names,
writes per‑dataset shards (.h5ad) and a single combined .h5ad.
Returns both the combined AnnData and an in‑memory AnnCollection.
"""

from pathlib import Path
import json
import scanpy as sc
import anndata as ad

from data_pipeline.src.config import DATASETS
from  data_pipeline.src.load_anndata import load_anndata


# Paths (always inside data_pipeline/data/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "pbmc_raw"
PROC_DIR     = DATA_DIR / "pbmc_processed"
SHARD_DIR    = PROC_DIR / "shards"


# Load all datasets
def load_all_datasets(meta_dict=DATASETS, raw_dir=RAW_DIR):
    """Load each dataset folder into an annotated AnnData."""
    adatas = []
    for name, meta in meta_dict.items():
        folder = raw_dir / name
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder} (run download_script.py first)")
        print(f"📦 Loading dataset: {name}")
        adata = load_anndata(folder, meta)
        adata.obs["dataset"] = name
        adatas.append(adata)
    return adatas


# Freeze label categories
def freeze_label_categories(adatas):
    """Unify category vocabularies and regenerate codes."""
    keys = [f"cell_type_lvl{i}" for i in range(1, 5)]
    label_maps = {}

    for key in keys:
        union = sorted(set().union(*[set(a.obs[key].cat.categories) for a in adatas]))
        label_maps[key] = union

    for a in adatas:
        for key, cats in label_maps.items():
            a.obs[key] = a.obs[key].cat.set_categories(cats)
            a.obs[f"{key}__code"] = a.obs[key].cat.codes.astype("int64")

    print("✓ Label categories frozen across datasets.")
    return adatas, label_maps


def harmonize_varnames(adatas):
    """Upper‑case and deduplicate gene names (optional)."""
    for i, a in enumerate(adatas):
        genes = a.var_names.astype(str)
        genes = genes.str.strip().str.upper()
        mask = ~genes.duplicated(keep="first")
        adatas[i] = a[:, mask].copy()
        adatas[i].var_names = genes[mask]
    print("✓ var_names harmonized.")
    return adatas


#  Write per‑dataset shards + combined file
def write_shards(adatas, out_dir=SHARD_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for a in adatas:
        name = str(a.obs["dataset"].unique()[0])
        out_path = out_dir / f"{name}.h5ad"
        print(f"Writing shard: {out_path}")
        a.write(out_path)
        paths.append(out_path)
    return paths


def combine_and_write(adatas, label_maps, out_file=PROC_DIR / "pbmc_combined.h5ad"):
    """Merge all AnnData objects into a single combined AnnData."""
    print("Combining all datasets (outer join on genes)...")
    combined = ad.concat(adatas, join="outer", label="dataset", merge="unique")
    combined.uns["label_maps"] = label_maps
    out_file.parent.mkdir(parents=True, exist_ok=True)
    combined.write(out_file)
    print(f"✓ Combined AnnData written to {out_file}")
    return combined


def write_label_map_json(label_maps, out_file=PROC_DIR / "label_maps.json"):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(label_maps, f, indent=2)
    print(f"✓ Label maps saved to {out_file}")



def make_collection_from_shards(shard_dir=SHARD_DIR):
    """Create an in‑memory AnnCollection over written shards."""
    from anndata.experimental import AnnCollection
    import anndata as ad
    paths = list(Path(shard_dir).glob("*.h5ad"))
    ads   = [ad.read_h5ad(p, backed="r") for p in paths]
    coll  = AnnCollection(ads, join_vars="outer", join_obs="outer", label="dataset")
    print(f"✓ AnnCollection constructed from {len(paths)} shards.")
    return coll


def run_combine(write_combined=True, do_write_shards=True, harmonize_var=False):
    print("Loading all PBMC AnnData objects...")
    adatas = load_all_datasets()

    print("Freezing label categories...")
    adatas, label_maps = freeze_label_categories(adatas)

    if harmonize_var:
        print("Harmonizing gene names...")
        adatas = harmonize_varnames(adatas)

    if do_write_shards:
        print("Writing per‑dataset shards...")
        write_shards(adatas)

    combined = None
    if write_combined:
        print("Writing combined dataset...")
        combined = combine_and_write(adatas, label_maps)
        write_label_map_json(label_maps)

    print("Building in‑memory AnnCollection...")
    collection = make_collection_from_shards(SHARD_DIR)

    print("\nCombine stage complete!\n")
    return combined, collection


