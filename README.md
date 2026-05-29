# Delta Expression Encoder

**A Transformer-based model for encoding psilocybin transcriptional perturbation signatures from single-nucleus RNA-seq data.**

Part of the [attention-to-bio](https://github.com/saijayakumar/attention-to-bio) project — a multimodal Transformer architecture for modeling individual differences in psilocybin therapeutic response.

> **Paper:** Jayakumar SK. *A Transformer-Based Delta Expression Encoder for Psilocybin Transcriptional Response: Architecture, Representations, and Biological Validation.* bioRxiv. 2026. doi:[@@FLAG: fill after posting]

---

## What this is

The delta expression encoder is a 4-layer Transformer trained to classify differential gene expression (DEG) status — upregulated, downregulated, or neutral — per gene per timepoint from pseudobulk single-nucleus RNA-seq profiles. It operates on the *perturbation signature* of a drug: given baseline expression and post-drug expression at one timepoint, what is each gene doing?

The model is trained on the [Liao et al. 2025](https://doi.org/10.1101/2025.01.04.631335) psilo-seq dataset (mouse medial frontal cortex, psilocybin and ketamine, 5 timepoints, 18 cell types, 623 pseudobulk examples) and learns biologically coherent representations without any pathway supervision.

### Key findings

- **Cell-type accuracy gradient:** Per-cell-type prediction difficulty inversely tracks psilocybin response magnitude across all 18 cell types (Spearman ρ = −0.76, p = 0.0003 vs. DEG count) — emergent from the data, not supervised. Directly validates Shao et al. 2025.
- **Downregulation stereotypy:** Psilocybin-induced transcriptional downregulation is significantly more stereotyped across individuals than upregulation (Mann-Whitney p < 0.0001), with a cortical depth gradient across excitatory subtypes. Novel finding.
- **Drug-specific co-regulation modules:** Attention analysis recovers drug-specific gene modules without pathway supervision — serotonin receptor and alkaloid response modules are psilocybin-dominant; NMDA receptor and structural plasticity modules are ketamine-dominant.
- **Population-level learning confirmed:** Individual animal identity is at chance in linear probing (balanced accuracy 0.100, chance ≈ 0.059), confirming the model learned population-level biology rather than memorizing individual animals.
- **Cross-dataset generalization (Bagot Lab, McGill):** Three-tier generalization structure across 11 cell types in an independent scRNA-seq dataset — full generalization in non-neuronal cells, magnitude-only generalization in excitatory subtypes, systematic failure in L5-6 NP attributable to dataset-specific co-regulation structure.

---

## Notebooks

> GitHub cannot render large notebooks inline. Use the nbviewer links below to view all figures and outputs.

| Notebook | Contents | View |
|----------|----------|------|
| `module2_prototype.ipynb` | End-to-end training pipeline, early representation analysis | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/saijaya/delta-expression-encoder/blob/main/module2_prototype.ipynb) |
| `module2_investigate_INV1-12.ipynb` | INV-1 through 12: PCA/UMAP, attention modules, probing, HTR2A, Bagot cross-dataset | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/saijaya/delta-expression-encoder/blob/main/module2_investigate_INV1-12.ipynb) |
| `module2_investigate_INV13-17.ipynb` | INV-13 through 17: wrong-prediction attention, variance asymmetry, biotype, magnitude ρ | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/saijaya/delta-expression-encoder/blob/main/module2_investigate_INV13-17.ipynb) |

---

## Repository structure

```
delta-expression-encoder/
├── README.md
├── module2_prototype.py                   # End-to-end training pipeline + early analysis
├── module2_investigate.py                 # Full investigation script — INV-1 through INV-17
├── module2_prototype.ipynb                # Notebook with embedded outputs
├── module2_investigate_INV1-12.ipynb      # Notebook with embedded outputs (INV-1–12)
├── module2_investigate_INV13-17.ipynb     # Notebook with embedded outputs (INV-13–17)
├── figures/
│   └── (generated figures, not tracked in git)
└── checkpoints/
    └── (model checkpoints, not tracked in git — see Zenodo)
```

---

## Model architecture

```
Input: 510 genes × 6 timepoints (pseudobulk expression matrix)
       Sentinel value −1.0 for unmeasured timepoints

Each token = expression projection (Linear → 128d)
           + gene identity embedding (Embedding[510, 128])
           + sinusoidal time encoding (fixed, 6 positions)
           + cell type conditioning (Embedding[18, 128])
           + drug conditioning (Embedding[2, 128])

Transformer encoder: 4 layers, 4 heads, d_ff=512, dropout=0.05
Full self-attention across all 3,060 tokens (510 genes × 6 timepoints)

Output: 4-class classification per token — DOWN / NEUTRAL / UP / BASELINE
Embeddings: mean-pool over all token representations → 128-dim vector

Total parameters: 862,084
```

The central design innovation is the **sentinel value strategy**: unmeasured timepoints are marked with −1.0 (outside the normalized expression range) and replaced by a learned mask token before the Transformer. This prevents the model from exploiting imputed values as a shortcut and forces it to attend only to real measurements. Combined with a **two-tier training curriculum** (individual animal examples + population mean examples), this enables learning from a between-subjects dataset where each animal contributes data at only one timepoint.

---

## Quickstart

### Requirements

```bash
pip install torch numpy pandas scanpy scipy scikit-learn matplotlib gprofiler-official
```

Python 3.9+. GPU strongly recommended (trained on NVIDIA L4; inference works on CPU for small batches).

### Load the pretrained model

```python
import torch
from model import DeltaExpressionEncoder

# Load checkpoint from Zenodo (see Data and checkpoints below)
ckpt = torch.load('delta_v5_v2_epoch50.pt', weights_only=False)

model = DeltaExpressionEncoder(
    n_genes=510,
    n_timepoints=6,
    n_cell_types=18,
    n_drugs=2
)
model.load_state_dict(ckpt['model_state'])
model.eval()
```

### Get embeddings

```python
import torch

# x: (batch, 510, 6) — expression matrix, sentinel −1.0 for unmeasured timepoints
# ct_ids: (batch,) — cell type indices
# drug_ids: (batch,) — drug indices (0=Psilo, 1=Ket)

x = torch.randn(1, 510, 6)        # replace with real pseudobulk data
ct_ids = torch.tensor([0])         # L2/3 IT
drug_ids = torch.tensor([0])       # Psilocybin

with torch.no_grad():
    embedding = model.get_embedding(x, ct_ids, drug_ids)
    # embedding: (1, 128)
```

### Run classification

```python
with torch.no_grad():
    out, logits = model(x, ct_ids, drug_ids)
    # logits: (1, 3060, 4) — per-token class logits
    # classes: 0=DOWN, 1=NEUTRAL, 2=UP, 3=BASELINE

    probs = torch.softmax(logits, dim=-1)
    # reshape to (1, 510, 6, 4) for gene × timepoint access
    probs = probs.reshape(1, 510, 6, 4)
```

### Training from scratch

```bash
# @@FLAG: fill in once train.py is cleaned and argparse is added
python train.py \
    --data_path /path/to/input_tensors_v5_v2.pt \
    --label_path /path/to/label_tensor_v5_v2.pt \
    --output_dir ./checkpoints \
    --epochs 275 \
    --lr 1e-3
```

---

## Data and checkpoints

| Artifact | Location | Notes |
|---|---|---|
| Liao et al. 2025 psilo-seq dataset (raw) | [SRA BioProject PRJNA1204073](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1204073) | Raw FASTQ |
| Liao et al. 2025 psilo-seq dataset (processed) | Zenodo [@@FLAG: fill DOI] | AnnData h5ad with Allen Institute cell-type annotations |
| Pretrained checkpoint (delta_v5_v2_epoch50.pt) | Zenodo [@@FLAG: fill your own DOI] | Best embedding structure (epoch 50). Use for representation analysis. |
| Epoch 275 checkpoint | Zenodo [@@FLAG: fill your own DOI] | Best classification accuracy (69.5% weighted). |
| Gene panel (510 genes) | Zenodo [@@FLAG: fill] | gene_set_500_v2.pt — list of 510 genes used for training |

The Bagot Lab cross-validation dataset (GSE283929) is available on [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE283929) upon publication of the primary Bagot et al. manuscript.

---

## Reproducing the paper figures

All figures in the paper are generated by the analysis scripts. After loading the model and data:

| Figure | Script | Key function |
|---|---|---|
| Fig 1 — PCA embedding (drug + cell type) | `analysis/investigate.py` | INV-1B block |
| Fig 2 — Cell-type distance matrix | `analysis/investigate.py` | INV-1C block |
| Fig 3 — Cell-type accuracy bar chart | `analysis/investigate.py` | Accuracy evaluation block |
| Fig 4 — DOWN/UP variance boxplot | `analysis/investigate.py` | Variance asymmetry block |
| Fig 5 — Co-attention heatmap | `analysis/investigate.py` | INV-5A/5B block |
| Fig 6 — Three-tier generalization | `analysis/investigate_bagot.py` | INV-17 block |
| Fig 7 — Architecture diagram | N/A | Created manually |

---

## Cell type index

| Index | Cell type | Class | Psilocybin sensitivity |
|---|---|---|---|
| 0 | L2/3 IT | Excitatory | Primary target (highest HTR2A) |
| 1 | L4/5 IT | Excitatory | Primary target |
| 2 | L5 ET | Excitatory | Secondary target |
| 3 | L5 IT | Excitatory | Secondary target |
| 4 | L5 NP | Excitatory | Low |
| 5 | L6 CT | Excitatory | Low |
| 6 | L6 IT | Excitatory | Primary target |
| 7 | L6b | Excitatory | Low |
| 8 | PV | GABAergic | Indirect (5-HT2C / 5-HT1A) |
| 9 | SST | GABAergic | Indirect |
| 10 | VIP | GABAergic | Indirect |
| 11 | Sncg | GABAergic | Indirect |
| 12 | Lamp5 | GABAergic | Indirect |
| 13 | Astro | Non-neuronal | Minimal |
| 14 | Micro | Non-neuronal | Minimal |
| 15 | Oligo | Non-neuronal | Minimal |
| 16 | OPC | Non-neuronal | Minimal |
| 17 | Endo | Non-neuronal | Minimal |

---

## Citation

If you use this model or code, please cite:

```bibtex
@article{jayakumar2026delta,
  title   = {A Transformer-Based Delta Expression Encoder for Psilocybin
             Transcriptional Response: Architecture, Representations,
             and Biological Validation},
  author  = {Jayakumar, Sai Krishna},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {@@FLAG: fill after bioRxiv posting}
}
```

Please also cite the underlying dataset:

```bibtex
@article{liao2025psilo,
  title   = {Single-nucleus transcriptomics reveals time-dependent and
             cell-type-specific effects of psilocybin on gene expression},
  author  = {Liao, C and O'Farrell, E and others},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.01.04.631335}
}
```

---

## Acknowledgments

Training data from the psilo-seq consortium (Liao et al. 2025, Kwan Lab, Cornell / University of Michigan). Special thanks to Ethan O'Farrell and Alex Kwan for providing the updated AnnData with animal ID metadata and for scientific guidance throughout model development.

This work was conducted as part of the Stanford BMDS program. Computational resources supported by Adobe Research.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Part of attention-to-bio

This repository contains Module 2 of the attention-to-bio project. The broader architecture includes:

- **Module 1** — Pharmacogenomic encoder (HTR2A, CYP2D6, ESM-2 receptor structural priors)
- **Module 2** — Delta expression encoder ← *this repo*
- **Module 3** — EEG neural dynamics encoder (LaBraM foundation model, in development)
- **Cross-modal bridge** — Fully connected cross-modal attention (planned)
