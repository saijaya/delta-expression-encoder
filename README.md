# Delta Expression Encoder

**A Transformer-based model for encoding psilocybin transcriptional perturbation signatures from single-nucleus RNA-seq data.**

Part of the [attention-to-bio](https://github.com/saijaya/attention-to-bio) project — a multimodal Transformer architecture for modeling individual differences in psilocybin therapeutic response.

> **Paper:** Jayakumar S. *A Transformer-Based Delta Expression Encoder for Psilocybin Transcriptional Response: Architecture, Representations, and Biological Validation.* bioRxiv. 2026. doi:[10.5281/zenodo.21095034](https://doi.org/10.5281/zenodo.21095034)

---

## What this is

The delta expression encoder is a 4-layer Transformer trained to classify differential gene expression (DEG) status — upregulated, downregulated, or neutral — per gene per timepoint from pseudobulk single-nucleus RNA-seq profiles. It operates on the *perturbation signature* of a drug: given baseline expression and post-drug expression at one timepoint, what is each gene doing?

The model is trained on the [Liao et al. 2025](https://doi.org/10.1101/2025.01.04.631335) psilo-seq dataset (mouse medial frontal cortex, psilocybin and ketamine, 5 timepoints, 18 cell types, 623 pseudobulk examples) and achieves 69.4% weighted classification accuracy while learning biologically coherent representations without any pathway supervision.

### Key findings

- **Biologically coherent representations:** Drug identity (1.000 ± 0.000), cell type (1.000 ± 0.000), and timepoint (0.974 ± 0.016) are perfectly or near-perfectly linearly decodable from frozen embeddings, while animal identity is at chance (0.100 ± 0.038). A probe for neuronal class (excitatory / inhibitory / non-neuronal — never provided as a training label) achieves 1.000 ± 0.000 balanced accuracy, confirming the model inferred biological taxonomy from expression patterns alone. Hierarchical clustering of cell-type embeddings recovers the known excitatory / inhibitory / non-neuronal taxonomy without supervision.

- **Downregulation stereotypy:** Psilocybin-induced transcriptional downregulation is significantly more stereotyped across individuals than upregulation (DOWN std=0.169 vs UP std=0.240; Mann-Whitney U=18615.0, p<0.0001), with a cortical depth gradient across excitatory subtypes (ratio range 1.07×–1.61×). Novel finding, not previously reported.

- **Cell-type accuracy gradient:** Per-cell-type classification accuracy inversely tracks psilocybin response magnitude (Spearman ρ=−0.885, p<0.001 vs. DEG count). L2/3 IT neurons — the primary HTR2A-expressing psilocybin target — show the lowest accuracy (28.3%); endothelial cells show the highest (99.6%). A direct test of the HTR2A-gating hypothesis returned a significant *negative* correlation (ρ=−0.7088, p=0.0021), inconsistent with simple receptor-dose dependence.

- **Drug-specific co-regulation modules:** Without pathway supervision, attention analysis recovers pharmacologically correct gene co-regulation modules. The primary ketamine-dominant cluster is anchored by *Grin2b* (GluN2B, ketamine's direct molecular target). The primary psilocybin-dominant cluster contains *Fos* and *Bdnf* — the canonical immediate-early gene and primary mediator of psilocybin-induced synaptic plasticity. The model separates *Grin2b* (ketamine-dominant) from *Grin2a* (psilocybin-dominant), reflecting known differences in NMDA subunit synaptic localization.

- **Attention ≠ DEG significance:** Top attention genes (by temporal CV) show zero overlap with Liao et al.'s top DEGs at 1h and 72h in L2/3 IT neurons (hypergeometric p=1.0, 0/30 overlap at both timepoints, below chance expectation). The model's attention priorities track a distinct signal from classical fold-change significance.

- **Temporal structure not recovered:** The model fails to recover psilocybin's biphasic temporal structure (peak-to-trough difference: 0.019 r units), attributable mechanistically to the between-subjects training data design. Documented as a limitation with quantitative bound.

---

## Repository structure

```
delta-expression-encoder/
├── README.md
├── module2_prototype.py          # Training pipeline — data loading, model, training loop
├── module2_investigate.py        # Analysis script — all paper figures and results
└── notebooks/
    ├── module2_prototype.ipynb   # Training notebook with embedded outputs
    └── module2_investigate.ipynb # Analysis notebook with embedded outputs
```

Model checkpoints and processed data are on Zenodo (see below) — not tracked in git.

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

Output: 4-class logits per token — DOWN / NEUTRAL / UP / BASELINE
Embeddings: mean-pool over final-layer token representations → 128-dim vector

Total parameters: 862,084
Training: AdamW, lr=1e-3, linear warmup 20 epochs, cosine decay to 300 epochs
```

The central design innovation is the **sentinel value strategy**: unmeasured timepoints are marked with −1.0 (outside the normalized expression range) and replaced by a learned mask token before the Transformer. This prevents the model from exploiting imputed values as a shortcut and forces it to attend only to real measurements. Combined with a **two-tier training curriculum** (individual animal examples + population mean examples), this enables learning from a between-subjects dataset where each animal contributes data at only one timepoint.

---

## Quickstart

### Requirements

```bash
pip install torch numpy pandas scanpy scipy scikit-learn matplotlib gprofiler-official umap-learn seaborn
```

Python 3.9+. GPU strongly recommended (trained on NVIDIA L4; inference works on CPU for small batches).

### Load the pretrained model

```python
import torch
import math
import torch.nn as nn

class DeltaExpressionEncoder(nn.Module):
    # Full definition in module2_investigate.py / module2_prototype.py
    ...

# Load checkpoint from Zenodo
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
# x: (batch, 510, 6) — expression matrix, sentinel −1.0 for unmeasured timepoints
# ct_ids: (batch,) — cell type index (see cell type table below)
# drug_ids: (batch,) — 0=Psilocybin, 1=Ketamine

x        = torch.randn(1, 510, 6)
ct_ids   = torch.tensor([3])   # 007 L2/3 IT CTX Glut
drug_ids = torch.tensor([0])   # Psilocybin

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
    probs = probs.reshape(1, 510, 6, 4)  # (batch, genes, timepoints, classes)
```

### Training from scratch

The full training pipeline is in `module2_prototype.py`. It requires the processed AnnData from Zenodo and auto-resumes from checkpoints. Run top-to-bottom in a Colab session with GPU.

---

## Data and checkpoints

| Artifact | Location |
|---|---|
| Liao et al. 2025 psilo-seq (raw FASTQ) | [SRA BioProject PRJNA1204073](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1204073) |
| Liao et al. 2025 psilo-seq (processed AnnData) | [Zenodo 10.5281/zenodo.19666128](https://doi.org/10.5281/zenodo.19666128) |
| Model checkpoints (all epochs) | [Zenodo 10.5281/zenodo.21095034](https://doi.org/10.5281/zenodo.21095034) |

Checkpoints include: `delta_v5_v2_epoch50.pt` (best embedding structure, used for representation analyses), `delta_v5_v2_final.pt` (epoch 300, 69.4% weighted accuracy, used for classification results), and intermediate epoch checkpoints at 25-epoch intervals.

---

## Reproducing paper figures

All figures are generated by `module2_investigate.py`. After loading the model and artifacts from Zenodo:

| Figure | Block in script | Description |
|---|---|---|
| Fig 1A | INV-1B | PCA embedding colored by drug and cell type |
| Fig 1B | INV-1D | Cell type dendrogram (Ward linkage) |
| Fig 2A | Block 8 | Classification confusion matrix |
| Fig 2B | Block 11 (prototype) | DOWN vs UP inter-individual variance |
| Fig 3A | Block 8 | Per-cell-type accuracy bar chart |
| Fig 3B | Block 6 (HTR2A) | Accuracy vs DEG magnitude scatter |
| Fig 3C | Block 7 | HTR2A expression vs drug-separation silhouette |
| Fig 4A | INV-5C | Gene-gene co-attention heatmap, 15 clusters |
| Fig 4B | INV-2C | Layer-by-layer attention to Htr2a |
| Fig 5 | Block 10 | Attention-DEG overlap (hypergeometric) |
| Fig 6 | Block 13 (prototype) | Htr1f expression trajectory |
| Fig 7 | Block 9 | Temporal grammar validation |
| Fig 8 | N/A | Architecture schematic (created manually) |
| Fig 9 | INV-1D | Cell type dendrogram |

---

## Cell type index

| Index | Cell type string | Class | Psilocybin sensitivity |
|---|---|---|---|
| 0 | `004 L6 IT CTX Glut` | Excitatory | Moderate |
| 1 | `005 L5 IT CTX Glut` | Excitatory | Secondary target |
| 2 | `006 L4/5 IT CTX Glut` | Excitatory | Primary target |
| 3 | `007 L2/3 IT CTX Glut` | Excitatory | **Primary target (highest HTR2A)** |
| 4 | `022 L5 ET CTX Glut` | Excitatory | Secondary target |
| 5 | `029 L6b CTX Glut` | Excitatory | Low |
| 6 | `030 L6 CT CTX Glut` | Excitatory | Low |
| 7 | `032 L5 NP CTX Glut` | Excitatory | Low |
| 8 | `046 Vip Gaba` | Inhibitory | Indirect |
| 9 | `047 Sncg Gaba` | Inhibitory | Indirect |
| 10 | `049 Lamp5 Gaba` | Inhibitory | Indirect |
| 11 | `052 Pvalb Gaba` | Inhibitory | Indirect |
| 12 | `053 Sst Gaba` | Inhibitory | Indirect |
| 13 | `319 Astro-TE NN` | Non-neuronal | Minimal |
| 14 | `326 OPC NN` | Non-neuronal | Minimal |
| 15 | `327 Oligo NN` | Non-neuronal | Minimal |
| 16 | `333 Endo NN` | Non-neuronal | Minimal |
| 17 | `334 Microglia NN` | Non-neuronal | Minimal |

Cell type strings include numeric prefixes from the Allen Institute annotation scheme used in the Liao et al. 2025 dataset.

---

## Citation

If you use this model or code, please cite:

```bibtex
@article{jayakumar2026delta,
  title   = {A Transformer-Based Delta Expression Encoder for Psilocybin
             Transcriptional Response: Architecture, Representations,
             and Biological Validation},
  author  = {Jayakumar, Sai},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.5281/zenodo.21095034}
}
```

Please also cite the underlying dataset:

```bibtex
@article{liao2025psilo,
  title   = {Single-nucleus transcriptomics reveals cell type-specific and
             time-dependent effects of psilocybin and ketamine on gene expression},
  author  = {Liao, C and O'Farrell, E and others},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.01.04.631335}
}
```

---

## Acknowledgments

Training data from the psilo-seq consortium (Liao et al. 2025, Kwan Lab, Cornell / University of Michigan). Special thanks to Ethan O'Farrell and Alex Kwan for providing the updated AnnData with animal ID metadata, for the recommendation to use PCA over UMAP for embedding visualization, for the technical caveat regarding the Htr1f SMART-Seq/MERFISH discrepancy, and for reviewing a draft of the paper.

This work was conducted as part of the Stanford Biomedical Data Science program.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Part of attention-to-bio

This repository contains Module 2 of the attention-to-bio project. The broader architecture includes:

- **Module 1** — Pharmacogenomic encoder (genomic variants, ESM-2 receptor structural priors, PK modeling)
- **Module 2** — Delta expression encoder ← *this repo*
- **Module 3** — EEG neural dynamics encoder (LaBraM backbone, in development)
- **Cross-modal bridge** — Fully connected cross-modal attention (planned)
