# =============================================================================
# attention-to-bio — Delta Expression Encoder: Analysis Notebook
#
# Loads trained model and all artifacts from Google Drive checkpoints.
# Runs all analyses reported in the paper — no retraining required.
# Run top to bottom in a fresh Colab session (GPU recommended).
#
# Blocks:
#   1. Setup and artifact loading
#   2. Representation analysis (PCA, probes, dendrogram)
#   3. Attention analysis (temporal CV, entropy, layer evolution)
#   4. Co-attention clustering (global + L2/3 IT-specific)
#   5. HTR2A-gating tests (cell-type level + individual level)
#   6. Accuracy vs DEG magnitude
#   7. Individual-level HTR2A vs embedding position
#   8. Temporal grammar validation
#   9. Attention-DEG hypergeometric test
#   10. Biological class probe
#   11. Non-neuronal attention entropy
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import math
import json
import sqlite3
import types
import collections

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import (mannwhitneyu, pearsonr, spearmanr,
                          hypergeom, entropy as shannon_entropy)

from google.colab import drive
drive.mount('/content/drive')

!pip install umap-learn gprofiler-official seaborn -q
import umap
import seaborn as sns
from gprofiler import GProfiler

# ── Paths and device ─────────────────────────────────────────────────────────
DRIVE_PATH = '/content/drive/MyDrive/attention-to-bio'
DB_PATH    = f'{DRIVE_PATH}/data/differential_expression_v2.sqlite'
os.makedirs(f'{DRIVE_PATH}/figures', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Style constants ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.titleweight':   'bold',
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9,
    'legend.frameon':     False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          False,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.dpi':        200,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
})

C_EXCIT   = '#C0392B'
C_INHIB   = '#2980B9'
C_NONNEUR = '#27AE60'
C_PSILO   = '#E74C3C'
C_KET     = '#3498DB'
C_POP     = '#1A1A2E'
C_TREND   = '#95A5A6'
C_ANNOT   = '#555555'
ALPHA     = 0.85

DOWN, NEUTRAL, UP, BASELINE = 0, 1, 2, 3
NOT_MEASURED = -1.0


# =============================================================================
# BLOCK 1: Model definition (must match training exactly)
# =============================================================================
class DeltaExpressionEncoder(nn.Module):
    """4-layer Transformer encoder for psilocybin DEG classification."""

    def __init__(self, n_genes, n_timepoints, n_cell_types, n_drugs,
                 n_classes=4, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, dropout=0.05):
        super().__init__()
        self.n_genes, self.n_timepoints, self.d_model = (
            n_genes, n_timepoints, d_model)
        self.input_proj          = nn.Linear(1, d_model)
        self.gene_embedding      = nn.Embedding(n_genes, d_model)
        self.register_buffer('time_encoding',
                             self._sinusoidal(n_timepoints, d_model))
        self.celltype_embedding  = nn.Embedding(n_cell_types, d_model)
        self.drug_embedding      = nn.Embedding(n_drugs, d_model)
        self.mask_token          = nn.Parameter(torch.randn(d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer         = nn.TransformerEncoder(enc_layer,
                                                         num_layers=n_layers)
        self.classification_head = nn.Linear(d_model, n_classes)
        self.input_norm          = nn.LayerNorm(d_model)
        self.dropout_layer       = nn.Dropout(dropout)

    def _sinusoidal(self, n, d):
        pe  = torch.zeros(n, d)
        pos = torch.arange(0, n).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() *
                        -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def normalize(self, x):
        flat = x.reshape(x.shape[0], -1)
        mu   = flat.mean(1, keepdim=True)
        std  = flat.std(1, keepdim=True) + 1e-6
        return ((flat - mu) / std).reshape(x.shape)

    def _build_tokens(self, x, ct_ids, drug_ids):
        B = x.shape[0]
        tokens = self.input_proj(self.normalize(x).reshape(B, -1).unsqueeze(-1))
        g_ids = torch.arange(self.n_genes, device=x.device)
        g_ids = g_ids.unsqueeze(1).expand(-1, self.n_timepoints).reshape(-1)
        t_ids = torch.arange(self.n_timepoints, device=x.device)
        t_ids = t_ids.unsqueeze(0).expand(self.n_genes, -1).reshape(-1)
        tokens = (tokens
                  + self.gene_embedding(g_ids)
                  + self.time_encoding[t_ids]
                  + (self.celltype_embedding(ct_ids) +
                     self.drug_embedding(drug_ids)).unsqueeze(1))
        return self.dropout_layer(self.input_norm(tokens))

    def forward(self, x, ct_ids, drug_ids, mask=None):
        tokens = self._build_tokens(x, ct_ids, drug_ids)
        if mask is not None:
            B = x.shape[0]
            mf = mask.reshape(B, -1)
            mt = self.mask_token.unsqueeze(0).unsqueeze(0).expand(
                B, tokens.shape[1], -1)
            tokens = torch.where(mf.unsqueeze(-1), mt, tokens)
        out    = self.transformer(tokens)
        logits = self.classification_head(out)
        return out, logits

    def get_embedding(self, x, ct_ids, drug_ids):
        out, _ = self.forward(x, ct_ids, drug_ids)
        return out.mean(dim=1)

    def get_attention_weights(self, x, ct_ids, drug_ids):
        """Returns list of (B, seq_len, seq_len) tensors, one per layer."""
        tokens = self._build_tokens(x, ct_ids, drug_ids)
        all_attn, x_curr = [], tokens
        for layer in self.transformer.layers:
            _, attn_w = layer.self_attn(
                x_curr, x_curr, x_curr,
                need_weights=True, average_attn_weights=True)
            all_attn.append(attn_w.detach().cpu())
            x_curr = layer(x_curr)
        return all_attn


# =============================================================================
# BLOCK 2: Load artifacts from Drive
# =============================================================================
gene_data       = torch.load(f'{DRIVE_PATH}/checkpoints/gene_set_500_v2.pt',
                              weights_only=False)
final_genes_500 = gene_data['final_genes']
print(f"Gene set: {len(final_genes_500)} genes")

td              = torch.load(f'{DRIVE_PATH}/checkpoints/input_tensors_v5_v2.pt',
                              weights_only=False)
input_tensor_v5 = td['input_tensor']
metadata_v5     = td['metadata']
celltype_to_idx = td['celltype_to_idx']
drug_to_idx     = td['drug_to_idx']
print(f"Input tensor: {input_tensor_v5.shape}")

label_tensor_v5 = torch.load(f'{DRIVE_PATH}/checkpoints/label_tensor_v5_v2.pt',
                               weights_only=False)
print(f"Label tensor: {label_tensor_v5.shape}")

# Dimensions — always from tensor
N_GENES        = input_tensor_v5.shape[1]
N_TIMEPOINTS   = input_tensor_v5.shape[2]
N_CELL_TYPES   = len(celltype_to_idx)
N_DRUGS        = len(drug_to_idx)
TIMEPOINT_LIST = ['0h', '1h', '2h', '4h', '24h', '72h']
DRUG_TPS       = ['1h', '2h', '4h', '24h', '72h']
print(f"N_GENES={N_GENES} | N_TIMEPOINTS={N_TIMEPOINTS} | "
      f"N_CELL_TYPES={N_CELL_TYPES} | N_DRUGS={N_DRUGS}")

# Pre-computed embeddings (epoch 50)
EMB_PATH = f'{DRIVE_PATH}/checkpoints/embeddings_v5_e50.npy'
embeddings = np.load(EMB_PATH) if os.path.exists(EMB_PATH) else None

# Metadata arrays for indexing
drugs_arr    = np.array([m['drug']         for m in metadata_v5])
celltypes_arr= np.array([m['cell_type']    for m in metadata_v5])
animals_arr  = np.array([m['animal']       for m in metadata_v5])
tps_arr      = np.array([m['timepoint']    for m in metadata_v5])
extypes_arr  = np.array([m['example_type'] for m in metadata_v5])
idx_to_ct    = {v: k for k, v in celltype_to_idx.items()}
idx_to_drug  = {v: k for k, v in drug_to_idx.items()}
ct_list      = sorted(celltype_to_idx.keys())


# =============================================================================
# BLOCK 3: Instantiate model + load checkpoint
# =============================================================================
model = DeltaExpressionEncoder(
    n_genes=N_GENES, n_timepoints=N_TIMEPOINTS,
    n_cell_types=N_CELL_TYPES, n_drugs=N_DRUGS).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

ckpt = torch.load(f'{DRIVE_PATH}/checkpoints/delta_v5_v2_epoch50.pt',
                   weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Loaded epoch 50 | loss={ckpt['losses'][-1]:.4f}")

# Compute embeddings if not loaded
if embeddings is None:
    print("Computing embeddings...")
    all_emb = []
    with torch.no_grad():
        for i in range(len(metadata_v5)):
            x    = input_tensor_v5[i].unsqueeze(0).to(device)
            x_in = x.clone(); x_in[x_in == -1.0] = 0.0
            ct   = torch.tensor([metadata_v5[i]['celltype_idx']]).to(device)
            dr   = torch.tensor([metadata_v5[i]['drug_idx']]).to(device)
            all_emb.append(model.get_embedding(x_in, ct, dr).cpu().numpy())
    embeddings = np.vstack(all_emb)
    np.save(EMB_PATH, embeddings)
    print(f"Embeddings computed: {embeddings.shape}")

print(f"Working with embeddings: {embeddings.shape}")


# =============================================================================
# BLOCK 4: Helper utilities
# =============================================================================
def get_mask(drug=None, cell_type=None, example_type=None, timepoint=None):
    mask = np.ones(len(metadata_v5), dtype=bool)
    if drug         is not None: mask &= (drugs_arr     == drug)
    if cell_type    is not None: mask &= (celltypes_arr == cell_type)
    if example_type is not None: mask &= (extypes_arr   == example_type)
    if timepoint    is not None: mask &= (tps_arr        == timepoint)
    return mask

DRUG_COLORS = {'Psilo': C_PSILO, 'Ket': C_KET, 'population': '#95A5A6'}
CT_PALETTE  = plt.cm.tab20.colors
CT_COLORS   = {ct: CT_PALETTE[i % 20] for i, ct in enumerate(ct_list)}
ind_mask    = get_mask(example_type='individual')


# =============================================================================
# INVESTIGATION BLOCK 1 — REPRESENTATION ANALYSIS
# =============================================================================

# ── INV-1A: Effective dimensionality ─────────────────────────────────────────
pca_full  = PCA(n_components=128)
pca_full.fit(embeddings)
explained  = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)
n_90       = np.argmax(cumulative >= 0.90) + 1
n_95       = np.argmax(cumulative >= 0.95) + 1

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, 33), explained[:32], color='steelblue', alpha=0.8)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Per-PC explained variance (first 32)')
axes[0].axvline(x=n_90, color='red', linestyle='--', alpha=0.7,
                label=f'90% at PC{n_90}')
axes[0].legend()
axes[1].plot(range(1, 129), cumulative, color='steelblue', linewidth=2)
axes[1].axhline(y=0.90, color='red',    linestyle='--', alpha=0.7, label='90%')
axes[1].axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
axes[1].set_xlabel('Number of PCs')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative explained variance')
axes[1].legend(); axes[1].set_xlim(1, 64)
plt.suptitle('Effective dimensionality of learned embedding space', y=1.01)
plt.tight_layout()
plt.savefig(f'{DRIVE_PATH}/figures/pca_explained_variance.png', dpi=150)
plt.show()
print(f"90% variance: {n_90} PCs | Effective rank ~{n_90}/128")

# ── INV-1B: PCA scatter — Figure 1A ─────────────────────────────────────────
pca_n90 = PCA(n_components=n_90)
emb_n90 = pca_n90.fit_transform(embeddings)

sil_drug_pca = silhouette_score(emb_n90[ind_mask], drugs_arr[ind_mask])
sil_ct_pca   = silhouette_score(emb_n90[ind_mask], celltypes_arr[ind_mask])
print(f"Drug silhouette ({n_90} PCs): {sil_drug_pca:.4f}")
print(f"Cell type silhouette ({n_90} PCs): {sil_ct_pca:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'wspace': 0.35})

ax = axes[0]
for drug, color in [(d, DRUG_COLORS[d]) for d in ['Psilo', 'Ket']]:
    m = ind_mask & (drugs_arr == drug)
    ax.scatter(emb_n90[m, 0], emb_n90[m, 1],
               c=color, s=28, alpha=0.70, linewidths=0, label=drug)
pop_mask = ~ind_mask
ax.scatter(emb_n90[pop_mask, 0], emb_n90[pop_mask, 1],
           c=C_POP, s=90, marker='*', zorder=5, linewidths=0,
           label='Population mean')
ax.set_xlabel(f'PC1 ({pca_n90.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_n90.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Drug separation')
ax.legend(fontsize=9)

ax = axes[1]
cts_uniq = sorted(set(celltypes_arr[ind_mask]))
cmap_ct  = dict(zip(cts_uniq, cm.tab20(np.linspace(0, 1, len(cts_uniq)))))
for ct in cts_uniq:
    m = ind_mask & (celltypes_arr == ct)
    short = ' '.join(ct.split()[1:3])
    ax.scatter(emb_n90[m, 0], emb_n90[m, 1],
               c=[cmap_ct[ct]], s=22, alpha=0.70, linewidths=0, label=short)
ax.set_xlabel(f'PC1 ({pca_n90.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_n90.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Cell-type separation')
ax.legend(fontsize=7.5, ncol=2, loc='upper right',
          markerscale=1.5, handlelength=0.8, columnspacing=0.5)

plt.suptitle(f'PCA of 128-d embeddings  ·  {n_90} PCs  ·  '
             f'drug sil={sil_drug_pca:.4f}  ·  cell-type sil={sil_ct_pca:.4f}',
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(f'{DRIVE_PATH}/figures/pca_drug_celltype.png')
plt.show()

# ── INV-1C: Cell type distance matrix ────────────────────────────────────────
centroids = {}
for ct in ct_list:
    m = ind_mask & (celltypes_arr == ct)
    if m.sum() > 0:
        centroids[ct] = emb_n90[m].mean(axis=0)

ct_names  = list(centroids.keys())
ct_matrix = np.stack([centroids[ct] for ct in ct_names])
dist_mat  = cdist(ct_matrix, ct_matrix, metric='euclidean')

ct_class      = ['Excitatory' if 'Glut' in ct
                 else 'Inhibitory' if 'Gaba' in ct
                 else 'Non-neuronal' for ct in ct_names]
class_colors  = {'Excitatory': C_EXCIT, 'Inhibitory': C_INHIB,
                 'Non-neuronal': C_NONNEUR}
short_labels  = [' '.join(ct.split()[:2]) for ct in ct_names]

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(dist_mat, cmap='viridis_r', aspect='auto')
ax.set_xticks(range(len(ct_names))); ax.set_yticks(range(len(ct_names)))
ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
ax.set_yticklabels(short_labels, fontsize=7)
plt.colorbar(im, ax=ax, label='Euclidean distance in PCA space')
ax.set_title(f'Cell type distance matrix ({n_90}-PC space)\n'
             f'Darker = more similar embedding', fontsize=11)
ax.legend(handles=[Patch(color=c, label=l) for l, c in class_colors.items()],
          loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(f'{DRIVE_PATH}/figures/celltype_distance_matrix_pca.png', dpi=150)
plt.show()

# ── INV-1D: Cell type dendrogram — Figure 1B ─────────────────────────────────
Z_ct = linkage(ct_matrix, method='ward')
fig, ax = plt.subplots(figsize=(14, 6))
dend = dendrogram(Z_ct, labels=short_labels, leaf_rotation=60,
                  leaf_font_size=8, ax=ax,
                  color_threshold=0.6 * max(Z_ct[:, 2]))
ax.set_title('Hierarchical clustering of cell type embeddings (Ward linkage)\n'
             'Branch lengths reflect PCA-space distances', fontsize=11)
ax.set_ylabel('Distance (Ward linkage)')
for label in ax.get_xticklabels():
    full = next((ct for ct in ct_names
                 if ct.startswith(label.get_text().split()[0])), None)
    if full:
        label.set_color(class_colors[ct_class[ct_names.index(full)]])
plt.tight_layout()
plt.savefig(f'{DRIVE_PATH}/figures/celltype_dendrogram.png', dpi=150)
plt.show()


# =============================================================================
# INVESTIGATION BLOCK 2 — ATTENTION ANALYSIS
# =============================================================================

# ── INV-2A: Temporal attention CV (top 30 genes) ─────────────────────────────
TARGET_CT   = '007 L2/3 IT CTX Glut'
TARGET_DRUG = 'Psilo'

def compute_temporal_attention_cv(cell_type, drug, model, device):
    """Per-gene attention CV using the population mean example."""
    idx = next(i for i, m in enumerate(metadata_v5)
               if m['cell_type'] == cell_type
               and m['drug'] == drug
               and m['example_type'] == 'population')
    x    = input_tensor_v5[idx].unsqueeze(0).to(device)
    x_in = x.clone(); x_in[x_in == -1.0] = 0.0
    ct   = torch.tensor([metadata_v5[idx]['celltype_idx']]).to(device)
    dr   = torch.tensor([metadata_v5[idx]['drug_idx']]).to(device)
    with torch.no_grad():
        all_attn = model.get_attention_weights(x_in, ct, dr)
    avg_attn = torch.stack(all_attn).mean(0)[0]
    results = []
    for gi, gene in enumerate(final_genes_500):
        tp_attns = np.array([
            avg_attn[:, gi * N_TIMEPOINTS + ti].mean().item()
            for ti in range(N_TIMEPOINTS)])
        cv = tp_attns.std() / (tp_attns.mean() + 1e-10)
        results.append({'gene': gene, 'tp_attns': tp_attns, 'cv': cv,
                        'peak_tp': TIMEPOINT_LIST[tp_attns.argmax()],
                        'peak_val': tp_attns.max()})
    return sorted(results, key=lambda x: -x['cv'])

attn_results = compute_temporal_attention_cv(TARGET_CT, TARGET_DRUG, model, device)
top30 = attn_results[:30]
print(f"Top 30 genes by temporal attention CV ({TARGET_CT}, {TARGET_DRUG}):")
for rank, r in enumerate(top30, 1):
    print(f"  {rank:2d}. {r['gene']:<15} CV={r['cv']:.4f}  "
          f"peak@{r['peak_tp']}={r['peak_val']:.6f}")

# ── INV-2B: Attention entropy per gene ───────────────────────────────────────
idx  = next(i for i, m in enumerate(metadata_v5)
            if m['cell_type'] == TARGET_CT and m['drug'] == TARGET_DRUG
            and m['example_type'] == 'population')
x_in = input_tensor_v5[idx].unsqueeze(0).to(device)
x_in_clean = x_in.clone(); x_in_clean[x_in_clean == -1.0] = 0.0
ct_t = torch.tensor([metadata_v5[idx]['celltype_idx']]).to(device)
dr_t = torch.tensor([metadata_v5[idx]['drug_idx']]).to(device)

with torch.no_grad():
    all_attn = model.get_attention_weights(x_in_clean, ct_t, dr_t)

avg_attn_np = torch.stack(all_attn).mean(0)[0].numpy()

gene_entropies = []
for gi, gene in enumerate(final_genes_500):
    ent = np.mean([
        shannon_entropy(avg_attn_np[gi * N_TIMEPOINTS + ti] + 1e-10)
        for ti in range(N_TIMEPOINTS)])
    gene_entropies.append((gene, ent))
gene_entropies.sort(key=lambda x: x[1])

print("\nLowest entropy genes (most focused attention):")
for gene, ent in gene_entropies[:10]:
    print(f"  {gene:<15}: {ent:.4f}")
print("Highest entropy genes (most diffuse attention):")
for gene, ent in gene_entropies[-10:]:
    print(f"  {gene:<15}: {ent:.4f}")

# ── INV-2C: Layer-by-layer attention evolution ───────────────────────────────
focal_gene = 'Htr2a'
focal_gi   = final_genes_500.index(focal_gene)
layer_attns = [a[0].numpy() for a in all_attn]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for li, layer_attn in enumerate(layer_attns):
    ax = axes[li]
    for ti, tp in enumerate(TIMEPOINT_LIST):
        pos      = focal_gi * N_TIMEPOINTS + ti
        tp_means = [layer_attn[gi * N_TIMEPOINTS:(gi + 1) * N_TIMEPOINTS, pos].mean()
                    for gi in range(min(50, N_GENES))]
        ax.bar(range(len(tp_means)), tp_means, alpha=0.5)
    ax.set_title(f'Layer {li + 1}')
    ax.set_xlabel('Gene index (first 50)')
    if li == 0: ax.set_ylabel(f'Avg incoming attention to {focal_gene}')
plt.suptitle(f'Layer-by-layer incoming attention to {focal_gene}', y=1.01)
plt.tight_layout()
plt.savefig(f'{DRIVE_PATH}/figures/layer_attention_evolution.png', dpi=150)
plt.show()


# =============================================================================
# INVESTIGATION BLOCK 3 — PROBING
# =============================================================================

# ── INV-3A: Linear probes ────────────────────────────────────────────────────
ind_emb   = embeddings[ind_mask]
ind_drugs = drugs_arr[ind_mask]
ind_cts   = celltypes_arr[ind_mask]
ind_anis  = animals_arr[ind_mask]
ind_tps   = tps_arr[ind_mask]

def run_probe(X, y, name, cv=5):
    le  = LabelEncoder()
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    scores = cross_val_score(clf, X, le.fit_transform(y), cv=cv,
                             scoring='balanced_accuracy')
    print(f"  {name:<35}: {scores.mean():.3f} ± {scores.std():.3f}  "
          f"(n_classes={len(np.unique(y))})")
    return scores

print("Linear probe results (5-fold CV, balanced accuracy):")
run_probe(ind_emb, ind_drugs, 'Drug (Psilo vs Ket)')
run_probe(ind_emb, ind_cts,   'Cell type (18 classes)')
run_probe(ind_emb, ind_tps,   'Timepoint (5 classes)')
run_probe(ind_emb, ind_anis,  'Animal identity (34 animals)')

# ── INV-3B: Biological class probe (never in training signal) ────────────────
BIO_CLASS = {
    '007 L2/3 IT CTX Glut': 'Excitatory',
    '006 L4/5 IT CTX Glut': 'Excitatory',
    '005 L5 IT CTX Glut':   'Excitatory',
    '022 L5 ET CTX Glut':   'Excitatory',
    '032 L5 NP CTX Glut':   'Excitatory',
    '004 L6 IT CTX Glut':   'Excitatory',
    '030 L6 CT CTX Glut':   'Excitatory',
    '029 L6b CTX Glut':     'Excitatory',
    '052 Pvalb Gaba':        'Inhibitory',
    '053 Sst Gaba':          'Inhibitory',
    '046 Vip Gaba':          'Inhibitory',
    '049 Lamp5 Gaba':        'Inhibitory',
    '047 Sncg Gaba':         'Inhibitory',
    '319 Astro-TE NN':       'NonNeuronal',
    '326 OPC NN':            'NonNeuronal',
    '327 Oligo NN':          'NonNeuronal',
    '333 Endo NN':           'NonNeuronal',
    '334 Microglia NN':      'NonNeuronal',
}

bio_labels = np.array([BIO_CLASS.get(m['cell_type'], 'Unknown') for m in metadata_v5
                       if m['example_type'] == 'individual'])
emb_ind    = embeddings[ind_mask]
meta_ind   = [m for m in metadata_v5 if m['example_type'] == 'individual']

print(f"Label distribution: {collections.Counter(bio_labels)}")

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
bio_scores = []
for train_idx, test_idx in cv_strat.split(emb_ind, bio_labels):
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(emb_ind[train_idx], bio_labels[train_idx])
    bio_scores.append(balanced_accuracy_score(
        bio_labels[test_idx], clf.predict(emb_ind[test_idx])))

print(f"\nBiological class probe (Excitatory/Inhibitory/NonNeuronal):")
print(f"  Balanced accuracy: {np.mean(bio_scores):.3f} ± {np.std(bio_scores):.3f}")
print(f"  Chance: 0.333")
print(f"  Lift: {np.mean(bio_scores)/0.333:.2f}×")


# =============================================================================
# INVESTIGATION BLOCK 4 — DRUG SWAP PERTURBATION
# =============================================================================
pop_examples = [i for i, m in enumerate(metadata_v5)
                if m['example_type'] == 'population'][:20]

shifts = []
for idx in pop_examples:
    x_orig = input_tensor_v5[idx].unsqueeze(0).to(device)
    x_in   = x_orig.clone(); x_in[x_in == -1.0] = 0.0
    ct_t   = torch.tensor([metadata_v5[idx]['celltype_idx']]).to(device)
    dr_t   = torch.tensor([metadata_v5[idx]['drug_idx']]).to(device)
    dr_swap= torch.tensor([1 - metadata_v5[idx]['drug_idx']]).to(device)

    with torch.no_grad():
        emb_orig = model.get_embedding(x_in, ct_t, dr_t).cpu().numpy()
        emb_swap = model.get_embedding(x_in, ct_t, dr_swap).cpu().numpy()

    shifts.append(np.linalg.norm(emb_orig - emb_swap))

# Within-drug and cross-drug distances
within = [np.linalg.norm(
    model.get_embedding(
        input_tensor_v5[i].unsqueeze(0).to(device).clone().clamp(min=0),
        torch.tensor([metadata_v5[i]['celltype_idx']]).to(device),
        torch.tensor([metadata_v5[i]['drug_idx']]).to(device)
    ).detach().cpu().numpy() -
    model.get_embedding(
        input_tensor_v5[j].unsqueeze(0).to(device).clone().clamp(min=0),
        torch.tensor([metadata_v5[j]['celltype_idx']]).to(device),
        torch.tensor([metadata_v5[j]['drug_idx']]).to(device)
    ).detach().cpu().numpy())
    for i, j in zip(pop_examples[:10], pop_examples[10:20])]

print(f"Drug swap shift: {np.mean(shifts):.2f} ± {np.std(shifts):.2f}")
print(f"Within-drug distance (reference): {np.mean(within):.2f}")


# =============================================================================
# INVESTIGATION BLOCK 5 — CO-ATTENTION CLUSTERING
# =============================================================================

# ── INV-5A: Extract global gene-gene attention matrix ────────────────────────
def extract_gene_gene_attention(model, device, metadata_v5, input_tensor_v5,
                                 n_genes, n_timepoints, example_type='population'):
    pop_indices = [i for i, m in enumerate(metadata_v5)
                   if m['example_type'] == example_type]
    gene_attn_sum = np.zeros((n_genes, n_genes), dtype=np.float64)
    for count, idx in enumerate(pop_indices):
        x    = input_tensor_v5[idx].unsqueeze(0).to(device)
        x_in = x.clone(); x_in[x_in == -1.0] = 0.0
        ct   = torch.tensor([metadata_v5[idx]['celltype_idx']]).to(device)
        dr   = torch.tensor([metadata_v5[idx]['drug_idx']]).to(device)
        with torch.no_grad():
            all_attn = model.get_attention_weights(x_in, ct, dr)
        avg_attn = torch.stack(all_attn).mean(0)[0].numpy()
        T, G = n_timepoints, n_genes
        gene_attn_sum += avg_attn.reshape(G, T, G, T).sum(axis=(1, 3))
        if (count + 1) % 10 == 0:
            print(f"  Processed {count+1}/{len(pop_indices)} examples")
    gene_attn_agg = gene_attn_sum / len(pop_indices)
    gene_attn_sym = (gene_attn_agg + gene_attn_agg.T) / 2.0
    print(f"Gene-gene attention: {gene_attn_sym.shape}  mean={gene_attn_sym.mean():.6f}")
    return gene_attn_agg, gene_attn_sym

# Load or compute
ATTN_PATH = f'{DRIVE_PATH}/checkpoints/gene_gene_attn_raw.npy'
if os.path.exists(ATTN_PATH):
    gene_attn_raw = np.load(ATTN_PATH)
    gene_attn_sym = np.load(f'{DRIVE_PATH}/checkpoints/gene_gene_attn_sym.npy')
    print(f"Loaded attention matrix: {gene_attn_raw.shape}")
else:
    gene_attn_raw, gene_attn_sym = extract_gene_gene_attention(
        model, device, metadata_v5, input_tensor_v5, N_GENES, N_TIMEPOINTS)
    np.save(ATTN_PATH, gene_attn_raw)
    np.save(f'{DRIVE_PATH}/checkpoints/gene_gene_attn_sym.npy', gene_attn_sym)

# ── INV-5B: Cluster (Ward linkage, 15 clusters) ──────────────────────────────
gene_attn_z  = (gene_attn_raw - gene_attn_raw.mean()) / (gene_attn_raw.std() + 1e-8)
threshold    = np.percentile(gene_attn_z, 80)
gene_sparse  = gene_attn_z.copy(); gene_sparse[gene_sparse < threshold] = 0.0

g_sym  = (gene_sparse + gene_sparse.T) / 2.0
g_max  = g_sym.max(axis=1, keepdims=True); g_max[g_max == 0] = 1.0
g_norm = (g_sym / g_max + (g_sym / g_max).T) / 2.0
g_dist = np.clip(1.0 - g_norm, 0, None)
np.fill_diagonal(g_dist, 0.0)
g_dist = (g_dist + g_dist.T) / 2.0

N_CLUSTERS    = 15
Z_genes       = linkage(squareform(g_dist), method='ward')
cluster_labels = fcluster(Z_genes, N_CLUSTERS, criterion='maxclust')

print(f"Cluster sizes (N_CLUSTERS={N_CLUSTERS}):")
for c in range(1, N_CLUSTERS + 1):
    members = [final_genes_500[i] for i, lbl in enumerate(cluster_labels) if lbl == c]
    print(f"  C{c:2d} ({len(members):3d}): {', '.join(members[:8])}")

# ── INV-5C: Drug-specific P/K ratios ─────────────────────────────────────────
def extract_subset_attention(indices):
    attn_sum = np.zeros((N_GENES, N_GENES))
    for idx in indices:
        x    = input_tensor_v5[idx].unsqueeze(0).to(device)
        x_in = x.clone(); x_in[x_in == -1.0] = 0.0
        ct   = torch.tensor([metadata_v5[idx]['celltype_idx']]).to(device)
        dr   = torch.tensor([metadata_v5[idx]['drug_idx']]).to(device)
        with torch.no_grad():
            all_attn = model.get_attention_weights(x_in, ct, dr)
        avg = torch.stack(all_attn).mean(0)[0].numpy()
        attn_sum += avg.reshape(N_GENES, N_TIMEPOINTS, N_GENES, N_TIMEPOINTS).sum(axis=(1, 3))
    return attn_sum / len(indices)

psilo_pops   = [i for i, m in enumerate(metadata_v5)
                if m['example_type'] == 'population' and m['drug'] == 'Psilo']
ket_pops     = [i for i, m in enumerate(metadata_v5)
                if m['example_type'] == 'population' and m['drug'] == 'Ket']
psilo_attn   = extract_subset_attention(psilo_pops)
ket_attn     = extract_subset_attention(ket_pops)

print("\nGlobal P/K ratio per cluster:")
print(f"{'C':>3} {'N':>5} {'Psilo':>10} {'Ket':>10} {'P/K':>7}  Top genes")
for c in range(1, N_CLUSTERS + 1):
    idxs = [i for i, lbl in enumerate(cluster_labels) if lbl == c]
    if len(idxs) < 2: continue
    p_coh = psilo_attn[np.ix_(idxs, idxs)].mean()
    k_coh = ket_attn[np.ix_(idxs, idxs)].mean()
    ratio = p_coh / (k_coh + 1e-10)
    members = [final_genes_500[i] for i in idxs]
    flag = ' ← PSILO' if ratio > 1.3 else (' ← KET' if ratio < 0.75 else '')
    print(f"  C{c:2d} ({len(idxs):3d}): {p_coh:.6f}  {k_coh:.6f}  {ratio:.3f}×{flag}")
    print(f"        {', '.join(members[:8])}")

# ── INV-5D: L2/3 IT-specific co-attention ────────────────────────────────────
l23_psilo_idx = [i for i, m in enumerate(metadata_v5)
                  if m['cell_type'] == '007 L2/3 IT CTX Glut'
                  and m['drug'] == 'Psilo' and m['example_type'] == 'individual']
l23_ket_idx   = [i for i, m in enumerate(metadata_v5)
                  if m['cell_type'] == '007 L2/3 IT CTX Glut'
                  and m['drug'] == 'Ket'   and m['example_type'] == 'individual']

def compute_gene_attn(indices):
    gene_attn = np.zeros((N_GENES, N_GENES))
    with torch.no_grad():
        for idx in indices:
            x  = input_tensor_v5[idx].unsqueeze(0).to(device)
            ct = torch.tensor([metadata_v5[idx]['celltype_idx']], dtype=torch.long).to(device)
            dr = torch.tensor([metadata_v5[idx]['drug_idx']],     dtype=torch.long).to(device)
            attn_agg = np.zeros((N_GENES, N_GENES))
            for layer_attn in model.get_attention_weights(x, ct, dr):
                a = layer_attn.squeeze(0).cpu().numpy()
                for ti in range(N_TIMEPOINTS):
                    for tj in range(N_TIMEPOINTS):
                        attn_agg += a[ti*N_GENES:(ti+1)*N_GENES,
                                      tj*N_GENES:(tj+1)*N_GENES]
            gene_attn += attn_agg
    return gene_attn / len(indices)

print(f"L2/3 IT Psilo: {len(l23_psilo_idx)}, Ket: {len(l23_ket_idx)}")
l23_psilo_attn = compute_gene_attn(l23_psilo_idx)
l23_ket_attn   = compute_gene_attn(l23_ket_idx)

l23_z      = (l23_psilo_attn - l23_psilo_attn.mean()) / (l23_psilo_attn.std() + 1e-8)
l23_sparse = l23_z.copy(); l23_sparse[l23_sparse < np.percentile(l23_z, 80)] = 0.0
l23_sym    = (l23_sparse + l23_sparse.T) / 2.0
l23_max    = l23_sym.max(axis=1, keepdims=True); l23_max[l23_max == 0] = 1.0
l23_norm   = (l23_sym / l23_max + (l23_sym / l23_max).T) / 2.0
l23_dist   = np.clip(1.0 - l23_norm, 0, None)
np.fill_diagonal(l23_dist, 0.0)
l23_dist   = (l23_dist + l23_dist.T) / 2.0
Z_l23      = linkage(squareform(l23_dist), method='ward')
l23_labels = fcluster(Z_l23, t=15, criterion='maxclust')

print("\nL2/3 IT-specific P/K ratio per cluster:")
for c in range(1, 16):
    idxs = [i for i, lbl in enumerate(l23_labels) if lbl == c]
    if len(idxs) < 2: continue
    p_coh  = l23_psilo_attn[np.ix_(idxs, idxs)].mean()
    k_coh  = l23_ket_attn[np.ix_(idxs, idxs)].mean()
    ratio  = p_coh / (k_coh + 1e-10)
    members = [final_genes_500[i] for i in idxs]
    flag   = ' ← PSILO' if ratio > 1.3 else (' ← KET' if ratio < 0.75 else '')
    print(f"  C{c:2d} ({len(idxs):3d}): P={p_coh:.6f}  K={k_coh:.6f}  ratio={ratio:.3f}×{flag}")
    print(f"        {', '.join(members[:8])}")

# ── INV-5E: Pathway enrichment per cluster ───────────────────────────────────
gp = GProfiler(return_dataframe=True)
cluster_enrichment = {}
print(f"\nPathway enrichment for {N_CLUSTERS} clusters (mmusculus)...")
for c in range(1, N_CLUSTERS + 1):
    members = [final_genes_500[i] for i, lbl in enumerate(cluster_labels) if lbl == c]
    if len(members) < 3:
        cluster_enrichment[c] = None; continue
    try:
        result = gp.profile(organism='mmusculus', query=members,
                            sources=['GO:BP', 'KEGG', 'REAC'],
                            significance_threshold_method='fdr',
                            user_threshold=0.05, no_evidences=False)
        cluster_enrichment[c] = result if not result.empty else None
        if not result.empty:
            top = result.nsmallest(3, 'p_value')
            print(f"C{c:2d} ({len(members):3d}):")
            for _, row in top.iterrows():
                print(f"  [{row['source']:8s}] {row['name'][:55]:55s}  p={row['p_value']:.2e}")
        else:
            print(f"C{c:2d} ({len(members):3d}): No significant enrichment")
    except Exception as e:
        print(f"C{c}: failed — {e}"); cluster_enrichment[c] = None


# =============================================================================
# INVESTIGATION BLOCK 6 — NON-NEURONAL ATTENTION ENTROPY
# =============================================================================
NON_NEURONAL = {'319 Astro-TE NN', '326 OPC NN', '327 Oligo NN',
                '333 Endo NN', '334 Microglia NN'}
EXCITATORY   = {'007 L2/3 IT CTX Glut', '006 L4/5 IT CTX Glut',
                '005 L5 IT CTX Glut', '022 L5 ET CTX Glut',
                '032 L5 NP CTX Glut', '004 L6 IT CTX Glut',
                '030 L6 CT CTX Glut', '029 L6b CTX Glut'}

nn_indices  = [i for i, m in enumerate(metadata_v5)
               if m['cell_type'] in NON_NEURONAL and m['example_type'] == 'individual']
exc_indices = [i for i, m in enumerate(metadata_v5)
               if m['cell_type'] in EXCITATORY   and m['example_type'] == 'individual']

def compute_attention_entropy(indices, n_sample=40):
    rng    = np.random.default_rng(42)
    sample = rng.choice(indices, size=min(n_sample, len(indices)), replace=False)
    entropies = []
    with torch.no_grad():
        for idx in sample:
            x  = input_tensor_v5[idx].unsqueeze(0).to(device)
            ct = torch.tensor([metadata_v5[idx]['celltype_idx']], dtype=torch.long).to(device)
            dr = torch.tensor([metadata_v5[idx]['drug_idx']],     dtype=torch.long).to(device)
            gene_attn = np.zeros(N_GENES)
            for layer_attn in model.get_attention_weights(x, ct, dr):
                a = layer_attn.squeeze(0).cpu().numpy()
                for ti in range(N_TIMEPOINTS):
                    gene_attn += a[:, ti * N_GENES:(ti + 1) * N_GENES].sum(axis=0)
            gene_attn = np.abs(gene_attn); gene_attn /= gene_attn.sum() + 1e-10
            entropies.append(shannon_entropy(gene_attn))
    return np.array(entropies)

print("Computing attention entropy...")
nn_entropies  = compute_attention_entropy(nn_indices)
exc_entropies = compute_attention_entropy(exc_indices)
max_h = np.log(N_GENES)

print(f"Non-neuronal: {nn_entropies.mean():.4f} ± {nn_entropies.std():.4f}  "
      f"({100*nn_entropies.mean()/max_h:.1f}% of max)")
print(f"Excitatory:   {exc_entropies.mean():.4f} ± {exc_entropies.std():.4f}  "
      f"({100*exc_entropies.mean()/max_h:.1f}% of max)")
stat, pval = mannwhitneyu(nn_entropies, exc_entropies, alternative='two-sided')
print(f"Mann-Whitney U={stat:.1f}, p={pval:.4f}")


# =============================================================================
# INVESTIGATION BLOCK 7 — HTR2A GATING TESTS
# =============================================================================

# Cell-type level: baseline HTR2A vs drug-separation silhouette
conn   = sqlite3.connect(DB_PATH)
htr2a_expr = pd.read_sql("""
    SELECT celltype, AVG(mean_expression) as htr2a_baseline
    FROM gene_expression
    WHERE gene_name = 'Htr2a' AND time = '0h' AND drug = 'none'
    GROUP BY celltype
""", conn)
conn.close()

sil_per_ct = {}
for ct in ct_list:
    m = ind_mask & (celltypes_arr == ct)
    if m.sum() < 4: continue
    try:
        s = silhouette_score(emb_n90[m], drugs_arr[m])
        sil_per_ct[ct] = s
    except Exception:
        pass

sil_df = pd.DataFrame([{'celltype': ct, 'silhouette': s}
                        for ct, s in sil_per_ct.items()])
merged = pd.merge(sil_df, htr2a_expr, on='celltype', how='inner')
rho_sil, p_sil = spearmanr(merged['htr2a_baseline'], merged['silhouette'])
print(f"\nHTR2A vs drug-silhouette: ρ={rho_sil:.4f}, p={p_sil:.4f}, n={len(merged)}")

# DEG magnitude correlation
conn = sqlite3.connect(DB_PATH)
psilo_response = pd.read_sql("""
    SELECT celltype, COUNT(*) as n_deg,
           AVG(ABS(log2FoldChange)) as mean_abs_lfc
    FROM de_results
    WHERE treatment_drug = 'Psilo' AND padj < 0.05
    AND ABS(log2FoldChange) >= 0.3
    GROUP BY celltype
""", conn)
conn.close()

merged2    = pd.merge(psilo_response, htr2a_expr, on='celltype', how='inner')
rho_n, p_n     = spearmanr(merged2['htr2a_baseline'], merged2['n_deg'])
rho_lfc, p_lfc = spearmanr(merged2['htr2a_baseline'], merged2['mean_abs_lfc'])
print(f"HTR2A vs n_deg:       ρ={rho_n:.4f}, p={p_n:.4f}")
print(f"HTR2A vs mean_abs_lfc: ρ={rho_lfc:.4f}, p={p_lfc:.4f}")


# =============================================================================
# INVESTIGATION BLOCK 8 — CONFUSION MATRIX AND PER-CELL-TYPE ACCURACY
# =============================================================================
from sklearn.metrics import confusion_matrix

all_true, all_pred = [], []
for i, meta in enumerate(metadata_v5):
    if meta['example_type'] != 'individual': continue
    x    = input_tensor_v5[i].unsqueeze(0).to(device)
    x_in = x.clone(); x_in[x_in == -1.0] = 0.0
    ct_t = torch.tensor([meta['celltype_idx']]).to(device)
    dr_t = torch.tensor([meta['drug_idx']]).to(device)
    with torch.no_grad():
        _, logits = model.forward(x_in, ct_t, dr_t)
    tp_idx   = TIMEPOINT_LIST.index(meta['timepoint'])
    n_genes  = label_tensor_v5[i].shape[0]
    tok_pos  = torch.arange(n_genes) * N_TIMEPOINTS + tp_idx
    y_pred   = logits[0][tok_pos].argmax(-1).cpu()
    y_true   = label_tensor_v5[i][:, tp_idx]
    mask     = y_true != BASELINE
    all_true.extend(y_true[mask].tolist())
    all_pred.extend(y_pred[mask].tolist())

cm_raw  = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
class_names_3 = ['DOWN', 'NEUTRAL', 'UP']

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, data, fmt, title in [
    (axes[0], cm_raw,  'd',    'Raw counts'),
    (axes[1], cm_norm, '.2f',  'Row-normalized (recall per class)'),
]:
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names_3, yticklabels=class_names_3, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True label'); ax.set_title(title)
plt.suptitle('DEG classification confusion matrix', y=1.03)
plt.tight_layout()
plt.savefig(f'{DRIVE_PATH}/figures/confusion_matrix.png', dpi=200)
plt.show()
print("Row-normalized:")
for i, cls in enumerate(class_names_3):
    print(f"  {cls}: DOWN={cm_norm[i,0]:.3f}  NEUTRAL={cm_norm[i,1]:.3f}  UP={cm_norm[i,2]:.3f}")


# =============================================================================
# INVESTIGATION BLOCK 9 — TEMPORAL GRAMMAR VALIDATION
# =============================================================================
def get_signed_scores(model, device, metadata_v5, input_tensor_v5,
                      cell_type, drug, n_genes, n_timepoints):
    idx  = next(i for i, m in enumerate(metadata_v5)
                if m['cell_type'] == cell_type and m['drug'] == drug
                and m['example_type'] == 'population')
    x    = input_tensor_v5[idx].unsqueeze(0).to(device)
    ct_t = torch.tensor([metadata_v5[idx]['celltype_idx']]).to(device)
    dr_t = torch.tensor([metadata_v5[idx]['drug_idx']]).to(device)
    with torch.no_grad():
        _, logits = model.forward(x, ct_t, dr_t)
    probs  = F.softmax(logits[0], dim=-1).reshape(n_genes, n_timepoints, 4).cpu().numpy()
    signed = probs[:, :, UP] - probs[:, :, DOWN]
    return signed[:, 1:]  # drug timepoints only

psilo_scores = get_signed_scores(model, device, metadata_v5, input_tensor_v5,
                                  TARGET_CT, 'Psilo', N_GENES, N_TIMEPOINTS)
ket_scores   = get_signed_scores(model, device, metadata_v5, input_tensor_v5,
                                  TARGET_CT, 'Ket',   N_GENES, N_TIMEPOINTS)

print("TEST 1: Psilo vs Ket score correlation per timepoint (Liao Fig 6c):")
tp_correlations = {}
for ti, tp in enumerate(DRUG_TPS):
    r, p = pearsonr(psilo_scores[:, ti], ket_scores[:, ti])
    tp_correlations[tp] = r
    print(f"  {tp}: r={r:.4f}, p={p:.4e}")

THRESHOLD   = 0.1
psilo_fracs = [(np.abs(psilo_scores[:, ti]) > THRESHOLD).mean() for ti in range(5)]
ket_fracs   = [(np.abs(ket_scores[:, ti])   > THRESHOLD).mean() for ti in range(5)]
print("\nTEST 2: Directional fraction per timepoint (Liao Fig 5b):")
for tp, pf, kf in zip(DRUG_TPS, psilo_fracs, ket_fracs):
    print(f"  {tp}: psilo={pf:.4f}  ket={kf:.4f}")

t1, t24, t72 = [tp_correlations[k] for k in ['1h', '24h', '72h']]
peak_to_trough = max(tp_correlations.values()) - min(tp_correlations.values())
print(f"\nPeak-to-trough difference: {peak_to_trough:.4f}")
print(f"Biphasic pattern: {psilo_fracs[0] > psilo_fracs[2] and psilo_fracs[4] > psilo_fracs[2]}")


# =============================================================================
# INVESTIGATION BLOCK 10 — ATTENTION-DEG HYPERGEOMETRIC TEST
# =============================================================================
top30_attn_genes = set(r['gene'] for r in attn_results[:30])

conn = sqlite3.connect(DB_PATH)
liao_1h = pd.read_sql("""
    SELECT gene_name FROM de_results
    WHERE celltype = '007 L2/3 IT CTX Glut' AND treatment_drug = 'Psilo'
    AND treatment_time = '1h' AND padj < 0.05
    ORDER BY padj ASC, ABS(log2FoldChange) DESC LIMIT 50
""", conn)
liao_72h = pd.read_sql("""
    SELECT gene_name FROM de_results
    WHERE celltype = '007 L2/3 IT CTX Glut' AND treatment_drug = 'Psilo'
    AND treatment_time = '72h' AND padj < 0.05
    ORDER BY padj ASC, ABS(log2FoldChange) DESC LIMIT 50
""", conn)
conn.close()

liao_1h_in_universe  = set(liao_1h['gene_name'])  & set(final_genes_500)
liao_72h_in_universe = set(liao_72h['gene_name']) & set(final_genes_500)
M, n = len(final_genes_500), 30

def hypergeom_test(attn_genes, liao_genes, label):
    k, N = len(attn_genes & liao_genes), len(liao_genes)
    exp  = n * N / M
    p    = hypergeom.sf(k - 1, M, n, N)
    print(f"\n  {label}: overlap={k}/{n}, expected={exp:.2f}, p={p:.4e}")
    return k, exp, p

k_1h,  exp_1h,  p_1h  = hypergeom_test(top30_attn_genes, liao_1h_in_universe,  '1h')
k_72h, exp_72h, p_72h = hypergeom_test(top30_attn_genes, liao_72h_in_universe, '72h')

# Sensitivity check
print("\nSensitivity (top 50/100 attention genes):")
for top_n in [50, 100]:
    attn_set = set(r['gene'] for r in attn_results[:top_n])
    o1  = attn_set & liao_1h_in_universe
    o72 = attn_set & liao_72h_in_universe
    p1  = hypergeom.sf(len(o1)  - 1, M, top_n, len(liao_1h_in_universe))
    p72 = hypergeom.sf(len(o72) - 1, M, top_n, len(liao_72h_in_universe))
    print(f"  top {top_n}: 1h overlap={len(o1)} (p={p1:.3e}), "
          f"72h overlap={len(o72)} (p={p72:.3e})")
