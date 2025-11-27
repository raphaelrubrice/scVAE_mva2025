#keep meta (URLs + cell type info)

# src/config.py
"""
PBMC lineage hierarchy (four levels)

lvl1 : stem vs non‑stem
lvl2 : major lymphoid lineage (B / NK / T)
lvl3 : T‑cell branch (CD4 / CD8; others repeat parent)
lvl4 : terminal subset (explicit marker‑based name for clarity)
"""

DATASETS = {
    # ---- Stem compartment ----------------------------------------------------
    "CD34": {
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd34/cd34_filtered_gene_bc_matrices.tar.gz",
        "lvl1": "Stem_cell",
        "lvl2": "CD34_HSC",
        "lvl3": "CD34_HSC",
        "lvl4": "CD34_HSC",
    },

    # ---- B and NK cells ------------------------------------------------------
    "CD19_B": {
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz",
        "lvl1": "Non_stem",
        "lvl2": "B_cell",
        "lvl3": "B_cell",
        "lvl4": "CD19_B_cell",
    },
    "CD56_NK": {
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd56_nk/cd56_nk_filtered_gene_bc_matrices.tar.gz",
        "lvl1": "Non_stem",
        "lvl2": "NK_cell",
        "lvl3": "NK_cell",
        "lvl4": "CD56_NK_cell",
    },

    # ---- CD4 T‑cell branch ---------------------------------------------------
    "CD4_helper": {
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd4_t_helper/cd4_t_helper_filtered_gene_bc_matrices.tar.gz",
        "lvl1": "Non_stem",
        "lvl2": "T_cell",
        "lvl3": "CD4_T",
        "lvl4": "CD4_CD3_T_helper",
    },
    "CD4_CD25": {  # regulatory T
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/regulatory_t/regulatory_t_filtered_gene_bc_matrices.tar.gz",
        "lvl1": "Non_stem",
        "lvl2": "T_cell",
        "lvl3": "CD4_T",
        "lvl4": "CD4_CD25_Treg",
    },
    "CD4_CD45RA_CD25neg": {  # naive T
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_t/naive_t_filtered_gene_bc_matrices.tar.gz",
        "lvl1": "Non_stem",
        "lvl2": "T_cell",
        "lvl3": "CD4_T",
        "lvl4": "CD4_CD45RA_CD25neg_naive_T",
    },
    "CD4_CD45RO": {  # memory T
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/memory_t/memory_t_filtered_gene_bc_matrices.tar.gz",
        "lvl1": "Non_stem",
        "lvl2": "T_cell",
        "lvl3": "CD4_T",
        "lvl4": "CD4_CD45RO_memory_T",
    },

    # ---- CD8 T‑cell branch ---------------------------------------------------
    "CD8": {  # cytotoxic T
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/cytotoxic_t/cytotoxic_t_filtered_gene_bc_matrices.tar.gz",
        "lvl1": "Non_stem",
        "lvl2": "T_cell",
        "lvl3": "CD8_T",
        "lvl4": "CD8_cytotoxic_T",
    },
    "CD8_CD45RA": {  # naive cytotoxic
        "url": "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_cytotoxic/naive_cytotoxic_filtered_gene_bc_matrices.tar.gz",
        "lvl1": "Non_stem",
        "lvl2": "T_cell",
        "lvl3": "CD8_T",
        "lvl4": "CD8_CD45RA_naive_T",
    },
}