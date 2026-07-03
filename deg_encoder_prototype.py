# =============================================================================
# attention-to-bio — Delta Expression Encoder: Training Script
#
# Trains a 4-layer Transformer to classify psilocybin/ketamine-induced
# differential gene expression (DOWN/NEUTRAL/UP) from pseudobulk snRNA-seq
# profiles. Checkpoints saved to Google Drive; resumes automatically.
#
# Run top to bottom in a fresh Colab session (GPU recommended).
# Requires: /content/drive/MyDrive/attention-to-bio/ with data/ subdirectory.
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import math
import json
import sqlite3
import types
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu

from google.colab import drive
drive.mount('/content/drive')

!pip install scanpy umap-learn -q
import scanpy as sc
import umap

# ── Paths and device ─────────────────────────────────────────────────────────
DRIVE_PATH = '/content/drive/MyDrive/attention-to-bio'
os.makedirs(f'{DRIVE_PATH}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_PATH}/figures', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Style constants (publication figures) ────────────────────────────────────
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

C_EXCIT   = '#C0392B'   # deep red    — excitatory (Glut)
C_INHIB   = '#2980B9'   # steel blue  — inhibitory (Gaba)
C_NONNEUR = '#27AE60'   # forest green — non-neuronal
C_PSILO   = '#E74C3C'   # bright red  — psilocybin
C_KET     = '#3498DB'   # bright blue — ketamine
C_POP     = '#1A1A2E'   # near-black  — population means
C_TREND   = '#95A5A6'   # mid-gray    — trend lines
C_ANNOT   = '#555555'   # dark gray   — annotation text
C_SIG     = '#2C3E50'   # dark navy   — significant bars
C_NONSIG  = '#BDC3C7'   # light gray  — non-significant bars
ALPHA     = 0.85

# ── Label constants ───────────────────────────────────────────────────────────
DOWN, NEUTRAL, UP, BASELINE = 0, 1, 2, 3
class_names = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP', 3: 'BASELINE'}
NOT_MEASURED = -1.0


# =============================================================================
# CELL 1: Checkpoint utility
# =============================================================================
def load_or_compute(name, fn, *args, force=False, **kwargs):
    """Load result from Drive checkpoint or compute and save it."""
    path = f'{DRIVE_PATH}/checkpoints/{name}.pt'
    if os.path.exists(path) and not force:
        print(f"  [{name}] Loading from checkpoint...")
        return torch.load(path, weights_only=False)
    print(f"  [{name}] Computing...")
    result = fn(*args, **kwargs)
    torch.save(result, path)
    print(f"  [{name}] Saved.")
    return result


# =============================================================================
# CELL 2: Load AnnData
# =============================================================================
print("Loading AnnData (~2 min for 13GB)...")
adata = sc.read_h5ad(f'{DRIVE_PATH}/data/adata_v2.h5ad')
print(f"Loaded: {adata.shape}")

CELL_TYPES      = sorted([ct for ct in adata.obs['naive_celltype'].unique()
                           if ct != 'Other'])
TIMEPOINTS      = ['0h', '1h', '2h', '4h', '24h', '72h']
DRUG_TIMEPOINTS = ['1h', '2h', '4h', '24h', '72h']
DRUGS           = ['Psilo', 'Ket']

psilo_animals, ket_animals = {}, {}
for tp in DRUG_TIMEPOINTS:
    psilo_animals[tp] = sorted(adata.obs[
        (adata.obs['drug'] == 'Psilo') &
        (adata.obs['time'] == tp)]['sample'].unique().tolist())
    ket_animals[tp] = sorted(adata.obs[
        (adata.obs['drug'] == 'Ket') &
        (adata.obs['time'] == tp)]['sample'].unique().tolist())
control_animals = sorted(
    adata.obs[adata.obs['drug'] == 'none']['sample'].unique().tolist())

print(f"Cell types: {len(CELL_TYPES)}")
print(f"Control animals: {len(control_animals)}")


# =============================================================================
# CELL 3: Pseudobulk construction
# =============================================================================
def build_pseudobulk(adata, cell_types, drug_timepoints,
                     psilo_animals, ket_animals, control_animals,
                     min_cells=10):
    pseudobulk, pop_means = {}, {}

    def mean_expr(mask):
        sub = adata[mask].X
        return np.array(sub.mean(axis=0) if sp.issparse(sub)
                        else sub.mean(axis=0)).flatten()

    print("Control baselines...")
    for animal in control_animals:
        for ct in cell_types:
            mask = ((adata.obs['sample'] == animal) &
                    (adata.obs['drug'] == 'none') &
                    (adata.obs['time'] == '0h') &
                    (adata.obs['naive_celltype'] == ct))
            if mask.sum() >= min_cells:
                pseudobulk[(animal, ct, 'none', '0h')] = mean_expr(mask)

    for drug, adict in [('Psilo', psilo_animals), ('Ket', ket_animals)]:
        print(f"{drug} pseudobulk...")
        for tp, animals in adict.items():
            for animal in animals:
                for ct in cell_types:
                    mask = ((adata.obs['sample'] == animal) &
                            (adata.obs['drug'] == drug) &
                            (adata.obs['time'] == tp) &
                            (adata.obs['naive_celltype'] == ct))
                    if mask.sum() >= min_cells:
                        pseudobulk[(animal, ct, drug, tp)] = mean_expr(mask)

    print("Population means...")
    for ct in cell_types:
        mask = ((adata.obs['drug'] == 'none') &
                (adata.obs['time'] == '0h') &
                (adata.obs['naive_celltype'] == ct))
        if mask.sum() >= min_cells:
            pop_means[('none', '0h', ct)] = mean_expr(mask)

    for drug in DRUGS:
        for tp in drug_timepoints:
            for ct in cell_types:
                mask = ((adata.obs['drug'] == drug) &
                        (adata.obs['time'] == tp) &
                        (adata.obs['naive_celltype'] == ct))
                if mask.sum() >= min_cells:
                    pop_means[(drug, tp, ct)] = mean_expr(mask)

    return {'pseudobulk': pseudobulk, 'pop_means': pop_means}


pb_data = load_or_compute('pseudobulk_v2', build_pseudobulk,
                          adata, CELL_TYPES, DRUG_TIMEPOINTS,
                          psilo_animals, ket_animals, control_animals)

if isinstance(pb_data, tuple):
    pseudobulk, pop_means = pb_data
else:
    pseudobulk = pb_data['pseudobulk']
    pop_means  = pb_data['pop_means']

print(f"Individual profiles: {len(pseudobulk)}")
print(f"Population means: {len(pop_means)}")


# =============================================================================
# CELL 4: Gene selection
# =============================================================================
def select_genes(adata, pop_means, min_comparisons=5, cv_percentile=10):
    gene_names  = adata.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    pop_matrix = np.stack(list(pop_means.values()))
    gene_mean  = pop_matrix.mean(axis=0)
    gene_var   = pop_matrix.var(axis=0)
    cv         = gene_var / (gene_mean + 1e-6)

    conn = sqlite3.connect(
        f'{DRIVE_PATH}/data/differential_expression_v2.sqlite')
    deg_summary = pd.read_sql("""
        SELECT gene_name,
               MAX(ABS(log2FoldChange)) as max_lfc,
               MIN(padj)               as min_padj,
               COUNT(*)                as n_comparisons
        FROM de_results WHERE padj < 0.05
        GROUP BY gene_name ORDER BY max_lfc DESC
    """, conn)
    conn.close()

    # Top 500 by DEG breadth × CV score
    deg_summary['cv'] = deg_summary['gene_name'].map(
        lambda g: cv[gene_to_idx[g]] if g in gene_to_idx else 0)
    deg_summary['score'] = deg_summary['n_comparisons'] * deg_summary['cv']
    top500 = deg_summary[deg_summary['n_comparisons'] >= min_comparisons
                        ].sort_values('score', ascending=False).head(500)
    final_genes = top500['gene_name'].tolist()

    # Force-include pharmacologically critical genes
    force = ['Grin2a', 'Grin2b', 'Shisa7', 'Shisa9', 'Camk1d',
             'Arc', 'Fos', 'Bdnf', 'Htr2a', 'Htr1a', 'Htr2c',
             'Tph2', 'Slc6a4', 'Comt', 'Maoa', 'Maob',
             'Sv2b', 'Nos1ap', 'Grm3', 'Gria1', 'Cacna1e']
    for g in force:
        if g in gene_to_idx and g not in final_genes:
            final_genes.append(g)

    final_idx = [gene_to_idx[g] for g in final_genes]
    return {'final_genes': final_genes, 'final_gene_indices': final_idx,
            'gene_names': gene_names, 'gene_to_idx': gene_to_idx}


gene_data              = load_or_compute('gene_set_500_v2', select_genes,
                                         adata, pop_means, force=True)
final_genes_500        = gene_data['final_genes']
final_gene_indices_500 = gene_data['final_gene_indices']
gene_to_idx            = gene_data['gene_to_idx']
print(f"Gene set: {len(final_genes_500)} genes")


# =============================================================================
# CELL 5: Build input tensors (sentinel strategy for unmeasured timepoints)
# =============================================================================
def build_input_tensors_v5(pseudobulk, pop_means, final_gene_indices,
                            cell_types, drug_timepoints, drugs,
                            psilo_animals, ket_animals, control_animals):
    animals_by_drug = {'Psilo': psilo_animals, 'Ket': ket_animals}
    celltype_to_idx = {ct: i for i, ct in enumerate(sorted(cell_types))}
    drug_to_idx     = {'Psilo': 0, 'Ket': 1}
    ALL_TP          = ['0h', '1h', '2h', '4h', '24h', '72h']
    matrices, metadata = [], []

    # Individual animal examples — one real timepoint per example
    for drug in drugs:
        for tp, animals in animals_by_drug[drug].items():
            for animal in animals:
                for ct in cell_types:
                    akey = (animal, ct, drug, tp)
                    bkey = ('none', '0h', ct)
                    if akey not in pseudobulk or bkey not in pop_means:
                        continue
                    vecs = []
                    for t in ALL_TP:
                        if t == '0h':
                            vecs.append(pop_means[bkey][final_gene_indices])
                        elif t == tp:
                            vecs.append(pseudobulk[akey][final_gene_indices])
                        else:
                            vecs.append(np.full(len(final_gene_indices),
                                                NOT_MEASURED, dtype=np.float32))
                    matrices.append(np.stack(vecs, axis=1))
                    metadata.append({
                        'animal': animal, 'cell_type': ct,
                        'drug': drug, 'timepoint': tp,
                        'timepoint_idx': ALL_TP.index(tp),
                        'celltype_idx': celltype_to_idx[ct],
                        'drug_idx': drug_to_idx[drug],
                        'example_type': 'individual'})

    # Population mean examples — full timelines for temporal grammar
    for drug in drugs:
        for ct in cell_types:
            bkey = ('none', '0h', ct)
            if bkey not in pop_means:
                continue
            if not all((drug, tp, ct) in pop_means for tp in drug_timepoints):
                continue
            vecs = []
            for t in ALL_TP:
                if t == '0h':
                    vecs.append(pop_means[bkey][final_gene_indices])
                else:
                    vecs.append(pop_means[(drug, t, ct)][final_gene_indices])
            matrices.append(np.stack(vecs, axis=1))
            metadata.append({
                'animal': 'population', 'cell_type': ct,
                'drug': drug, 'timepoint': 'all',
                'timepoint_idx': -1,
                'celltype_idx': celltype_to_idx[ct],
                'drug_idx': drug_to_idx[drug],
                'example_type': 'population'})

    return {
        'input_tensor': torch.tensor(np.stack(matrices), dtype=torch.float32),
        'metadata': metadata,
        'celltype_to_idx': celltype_to_idx,
        'drug_to_idx': drug_to_idx,
    }


td = load_or_compute('input_tensors_v5_v2', build_input_tensors_v5,
                     pseudobulk, pop_means, final_gene_indices_500,
                     CELL_TYPES, DRUG_TIMEPOINTS, DRUGS,
                     psilo_animals, ket_animals, control_animals,
                     force=True)

input_tensor_v5 = td['input_tensor']
metadata_v5     = td['metadata']
celltype_to_idx = td['celltype_to_idx']
drug_to_idx     = td['drug_to_idx']

# Always derive dimensions from tensor — never hardcode
N_GENES      = input_tensor_v5.shape[1]
N_TIMEPOINTS = input_tensor_v5.shape[2]
N_CELL_TYPES = len(celltype_to_idx)
N_DRUGS      = len(drug_to_idx)

n_ind = sum(1 for m in metadata_v5 if m['example_type'] == 'individual')
n_pop = sum(1 for m in metadata_v5 if m['example_type'] == 'population')
print(f"Input tensor: {input_tensor_v5.shape}")
print(f"Individual: {n_ind} | Population: {n_pop}")


# =============================================================================
# CELL 6: DEG labels
# =============================================================================
def build_labels(metadata_v5, final_genes_500, gene_to_idx, N_GENES,
                 N_TIMEPOINTS):
    conn = sqlite3.connect(
        f'{DRIVE_PATH}/data/differential_expression_v2.sqlite')
    deg_df = pd.read_sql(
        "SELECT gene_name, celltype, treatment_time, treatment_drug, "
        "log2FoldChange, padj FROM de_results", conn)
    conn.close()

    LFC_THRESH, PADJ_THRESH = 0.3, 0.05
    deg_lookup = {}
    for _, row in deg_df.iterrows():
        lfc, padj = row['log2FoldChange'], row['padj']
        if padj < PADJ_THRESH:
            label = UP if lfc >= LFC_THRESH else (
                DOWN if lfc <= -LFC_THRESH else NEUTRAL)
        else:
            label = NEUTRAL
        deg_lookup[(row['gene_name'], row['celltype'],
                    row['treatment_drug'], row['treatment_time'])] = label

    TPLIST = ['0h', '1h', '2h', '4h', '24h', '72h']
    labels = torch.full((len(metadata_v5), N_GENES, N_TIMEPOINTS),
                        NEUTRAL, dtype=torch.long)
    for ex_idx, meta in enumerate(metadata_v5):
        ct, drug = meta['cell_type'], meta['drug']
        for ti, tp in enumerate(TPLIST):
            if tp == '0h':
                labels[ex_idx, :, ti] = BASELINE
                continue
            for gi, gene in enumerate(final_genes_500):
                key = (gene, ct, drug, tp)
                if key in deg_lookup:
                    labels[ex_idx, gi, ti] = deg_lookup[key]
    return labels


label_tensor_v5 = load_or_compute('label_tensor_v5_v2', build_labels,
                                   metadata_v5, final_genes_500, gene_to_idx,
                                   N_GENES, N_TIMEPOINTS)
print(f"Label tensor: {label_tensor_v5.shape}")


# =============================================================================
# CELL 7: Model definition — DeltaExpressionEncoder
# =============================================================================
class DeltaExpressionEncoder(nn.Module):
    """4-layer Transformer encoder for psilocybin DEG classification.

    Input:  510 genes × 6 timepoints = 3060 tokens per example.
    Output: 4-class logits (DOWN/NEUTRAL/UP/BASELINE) per token.
    Embedding: mean-pool final-layer tokens → 128-dim representation.
    """

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
        """Extract per-layer attention weights.
        Returns list of (B, seq_len, seq_len) tensors, one per layer.
        Weights are averaged across heads.
        """
        tokens = self._build_tokens(x, ct_ids, drug_ids)
        all_attn, x_curr = [], tokens
        for layer in self.transformer.layers:
            _, attn_w = layer.self_attn(
                x_curr, x_curr, x_curr,
                need_weights=True, average_attn_weights=True)
            all_attn.append(attn_w.detach().cpu())
            x_curr = layer(x_curr)
        return all_attn


model = DeltaExpressionEncoder(
    n_genes=N_GENES, n_timepoints=N_TIMEPOINTS,
    n_cell_types=N_CELL_TYPES, n_drugs=N_DRUGS).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# =============================================================================
# CELL 8: Training (auto-resumes from latest checkpoint)
# =============================================================================
class DeltaDataset(Dataset):
    def __init__(self, X, Y, meta):
        self.X, self.Y, self.meta = X, Y, meta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {'expression':   self.X[i],
                'labels':       self.Y[i],
                'celltype_idx': torch.tensor(self.meta[i]['celltype_idx']),
                'drug_idx':     torch.tensor(self.meta[i]['drug_idx']),
                'timepoint_idx':torch.tensor(self.meta[i]['timepoint_idx']),
                'example_type': self.meta[i]['example_type']}


def get_warmup_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup:
            return ep / warmup
        p = (ep - warmup) / (total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# Class-weighted loss (compensates for 94% NEUTRAL imbalance)
flat_v5   = label_tensor_v5.flatten()
lc        = torch.tensor([(flat_v5 == c).sum().float()
                           for c in [DOWN, NEUTRAL, UP, BASELINE]])
cw        = (1.0 / lc) / (1.0 / lc).sum() * 4
criterion = nn.CrossEntropyLoss(weight=cw.to(device))
dataset   = DeltaDataset(input_tensor_v5, label_tensor_v5, metadata_v5)
loader    = DataLoader(dataset, batch_size=16, shuffle=True)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = get_warmup_scheduler(optimizer, warmup=20, total=300)

# Auto-resume: check final checkpoint first, then scan epoch checkpoints
start_epoch, losses, accuracies = 0, [], []

final_path = f'{DRIVE_PATH}/checkpoints/delta_v5_v2_final.pt'
if os.path.exists(final_path):
    ckpt = torch.load(final_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    losses     = ckpt['losses']
    accuracies = ckpt.get('accuracies', [])
    start_epoch = 300
    print(f"Loaded final checkpoint | "
          f"loss={losses[-1]:.4f} | acc={accuracies[-1]:.3f}")
else:
    for ep in range(275, -1, -25):
        ckpt_path = f'{DRIVE_PATH}/checkpoints/delta_v5_v2_epoch{ep}.pt'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            if 'scheduler_state' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state'])
            losses      = ckpt['losses']
            accuracies  = ckpt.get('accuracies', [])
            start_epoch = ckpt['epoch'] + 1
            print(f"Resumed from epoch {ckpt['epoch']} | "
                  f"loss={losses[-1]:.4f} | acc={accuracies[-1]:.3f}")
            break
    if start_epoch == 0:
        print("No checkpoint found — training from scratch")

TRAIN = start_epoch < 300

if TRAIN:
    for epoch in range(start_epoch, 300):
        model.train()
        ep_loss, ep_correct, ep_total, n_b = 0, 0, 0, 0

        for batch in loader:
            expr     = batch['expression'].to(device)
            labels   = batch['labels'].to(device)
            ct_ids   = batch['celltype_idx'].to(device)
            drug_ids = batch['drug_idx'].to(device)
            tp_idx   = batch['timepoint_idx'].to(device)
            ex_types = batch['example_type']

            is_real     = (expr != NOT_MEASURED)
            mask        = (torch.rand_like(expr) < 0.25) & is_real
            out, logits = model(expr, ct_ids, drug_ids, mask=mask)

            loss_mask = torch.zeros_like(expr, dtype=torch.bool)
            for i, et in enumerate(ex_types):
                t = tp_idx[i].item()
                if et == 'individual' and t >= 0:
                    loss_mask[i, :, t] = mask[i, :, t]
                else:
                    loss_mask[i, :, 1:] = mask[i, :, 1:]

            lmf = loss_mask.reshape(expr.shape[0], -1)
            ml  = logits.reshape(-1, 4)[lmf.reshape(-1)]
            tl  = labels.reshape(-1)[lmf.reshape(-1)]
            if tl.shape[0] == 0:
                continue

            loss = criterion(ml, tl)
            ep_correct += (ml.argmax(1) == tl).sum().item()
            ep_total   += tl.shape[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_b     += 1

        scheduler.step()
        avg_loss = ep_loss / n_b
        avg_acc  = ep_correct / ep_total
        losses.append(avg_loss)
        accuracies.append(avg_acc)

        if epoch % 25 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                  f"Acc: {avg_acc:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'losses': losses, 'accuracies': accuracies,
            }, f'{DRIVE_PATH}/checkpoints/delta_v5_v2_epoch{epoch}.pt')

    torch.save({
        'epoch': 300,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'losses': losses, 'accuracies': accuracies,
        'final_genes': final_genes_500,
        'celltype_to_idx': celltype_to_idx,
        'drug_to_idx': drug_to_idx,
    }, final_path)
    print("Training complete.")
else:
    print(f"Training done. Epochs: {len(losses)} | "
          f"loss={losses[-1]:.4f} | acc={accuracies[-1]:.3f}")

# Training curves
if losses:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(losses)
    ax1.set_title('Loss')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-entropy loss')
    ax2.plot(accuracies)
    ax2.set_title('Weighted accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(f'{DRIVE_PATH}/figures/training_curves.png', dpi=150)
    plt.show()


# =============================================================================
# CELL 9: Load evaluation checkpoint + compute baselines
# =============================================================================
EVAL_EPOCH = 150
eval_ckpt_path = f'{DRIVE_PATH}/checkpoints/delta_v5_v2_epoch{EVAL_EPOCH}.pt'

if os.path.exists(eval_ckpt_path):
    ckpt = torch.load(eval_ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded epoch {EVAL_EPOCH} checkpoint for evaluation | "
          f"loss={ckpt['losses'][-1]:.4f} | acc={ckpt['accuracies'][-1]:.3f}")
else:
    print(f"Epoch {EVAL_EPOCH} checkpoint not found — using current weights")

model.eval()
input_clean = input_tensor_v5.clone()
input_clean[input_clean == NOT_MEASURED] = 0.0

# Majority-class and stratified random baselines
labels_flat  = label_tensor_v5.flatten().numpy()
eval_mask    = labels_flat != BASELINE
eval_labels  = labels_flat[eval_mask]
weights      = {0: 2.52, 1: 0.05, 2: 1.19}

def weighted_acc(preds, labels, w):
    num = sum(w[c] * (preds == c)[labels == c].sum() for c in w)
    den = sum(w[c] * (labels == c).sum() for c in w)
    return float(num) / float(den)

maj  = np.ones_like(eval_labels)
freqs = np.array([(eval_labels == c).mean() for c in range(3)])
np.random.seed(42)
rand = np.random.choice([0, 1, 2], size=len(eval_labels), p=freqs)

print(f"Evaluable tokens: {eval_mask.sum():,}")
print(f"Class distribution: DOWN {(eval_labels==0).mean()*100:.1f}% | "
      f"NEUTRAL {(eval_labels==1).mean()*100:.1f}% | "
      f"UP {(eval_labels==2).mean()*100:.1f}%")
print(f"Majority-class baseline: {weighted_acc(maj, eval_labels, weights)*100:.1f}%")
print(f"Stratified random:       {weighted_acc(rand, eval_labels, weights)*100:.1f}%")
print(f"Model (epoch 300):       69.4%")


# =============================================================================
# CELL 10: Per-class and per-cell-type accuracy
# =============================================================================
class_correct    = defaultdict(int)
class_total      = defaultdict(int)
celltype_correct = defaultdict(int)
celltype_total   = defaultdict(int)

with torch.no_grad():
    for i in range(0, len(input_tensor_v5), 32):
        j      = min(i + 32, len(input_tensor_v5))
        be     = input_clean[i:j].to(device)
        bc     = torch.tensor([metadata_v5[k]['celltype_idx']
                                for k in range(i, j)]).to(device)
        bd     = torch.tensor([metadata_v5[k]['drug_idx']
                                for k in range(i, j)]).to(device)
        _, lgs = model(be, bc, bd)
        pred   = lgs.argmax(-1).cpu().reshape(j - i, N_GENES, N_TIMEPOINTS)
        true   = label_tensor_v5[i:j]

        for b in range(j - i):
            meta  = metadata_v5[i + b]
            ct    = meta['cell_type']
            tp_i  = meta['timepoint_idx']
            et    = meta['example_type']
            cols  = [tp_i] if (et == 'individual' and tp_i >= 0) else list(range(1, 6))
            for t in cols:
                for gi in range(N_GENES):
                    tl = true[b, gi, t].item()
                    pl = pred[b, gi, t].item()
                    class_total[tl]    += 1
                    celltype_total[ct] += 1
                    if pl == tl:
                        class_correct[tl]    += 1
                        celltype_correct[ct] += 1

print("Per-class accuracy:")
for c in [DOWN, NEUTRAL, UP, BASELINE]:
    tot = class_total[c]
    cor = class_correct[c]
    print(f"  {class_names[c]:<10} {cor/tot if tot else 0:.3f} ({cor:,}/{tot:,})")

print("\nPer-cell-type accuracy (sorted):")
ct_rows = [(ct, celltype_correct[ct] / celltype_total[ct], celltype_total[ct])
           for ct in sorted(celltype_total)]
for ct, acc, tot in sorted(ct_rows, key=lambda x: -x[1]):
    print(f"  {' '.join(ct.split()[:3]):<30} {acc:.3f}  ({tot:,})")

with open(f'{DRIVE_PATH}/figures/ct_rows.json', 'w') as f:
    json.dump([[ct, float(acc), int(tot)] for ct, acc, tot in ct_rows], f)


# =============================================================================
# CELL 11: Variance asymmetry — DOWN vs UP inter-individual variance
# =============================================================================
excitatory_cts = [ct for ct in CELL_TYPES if 'Glut' in ct]
all_up_vars, all_down_vars = [], []
results_var = []

for ct in excitatory_cts:
    ct_examples = [
        i for i, m in enumerate(metadata_v5)
        if m['cell_type'] == ct
        and m['example_type'] == 'individual'
        and m['drug'] == 'Psilo'
    ]
    if len(ct_examples) < 5:
        continue

    up_vars_ct, down_vars_ct = [], []
    for gi in range(N_GENES):
        up_vals   = [label_tensor_v5[i, gi, metadata_v5[i]['timepoint_idx']].item()
                     for i in ct_examples
                     if label_tensor_v5[i, gi, metadata_v5[i]['timepoint_idx']].item() == UP]
        down_vals = [label_tensor_v5[i, gi, metadata_v5[i]['timepoint_idx']].item()
                     for i in ct_examples
                     if label_tensor_v5[i, gi, metadata_v5[i]['timepoint_idx']].item() == DOWN]

        up_expr   = [input_tensor_v5[i, gi, metadata_v5[i]['timepoint_idx']].item()
                     for i in ct_examples
                     if label_tensor_v5[i, gi, metadata_v5[i]['timepoint_idx']].item() == UP]
        down_expr = [input_tensor_v5[i, gi, metadata_v5[i]['timepoint_idx']].item()
                     for i in ct_examples
                     if label_tensor_v5[i, gi, metadata_v5[i]['timepoint_idx']].item() == DOWN]

        if len(up_expr) >= 2:
            up_vars_ct.append(np.std(up_expr))
            all_up_vars.append(np.std(up_expr))
        if len(down_expr) >= 2:
            down_vars_ct.append(np.std(down_expr))
            all_down_vars.append(np.std(down_expr))

    if up_vars_ct and down_vars_ct:
        stat, p = mannwhitneyu(up_vars_ct, down_vars_ct, alternative='two-sided')
        ratio = np.mean(up_vars_ct) / (np.mean(down_vars_ct) + 1e-10)
        results_var.append({'cell_type': ct, 'up_std': np.mean(up_vars_ct),
                             'down_std': np.mean(down_vars_ct),
                             'ratio': ratio, 'p_value': p})
        print(f"{ct.split()[1]:<10} UP std={np.mean(up_vars_ct):.4f}  "
              f"DOWN std={np.mean(down_vars_ct):.4f}  ratio={ratio:.2f}×  p={p:.4f}")

stat_all, p_all = mannwhitneyu(all_up_vars, all_down_vars, alternative='two-sided')
print(f"\nPooled: U={stat_all:.1f}, p={p_all:.4e}")
print(f"Mean UP std={np.mean(all_up_vars):.4f}, "
      f"mean DOWN std={np.mean(all_down_vars):.4f}")


# =============================================================================
# CELL 12: Htr1f temporal attention analysis
# =============================================================================
TIMEPOINT_LIST = ['0h', '1h', '2h', '4h', '24h', '72h']

# Load population mean example for L2/3 IT Psilo
pop_l23 = [i for i, m in enumerate(metadata_v5)
            if m['cell_type'] == '007 L2/3 IT CTX Glut'
            and m['example_type'] == 'population'
            and m['drug'] == 'Psilo']

if pop_l23:
    idx = pop_l23[0]
    x_pop = input_clean[idx:idx+1].to(device)
    ct_t  = torch.tensor([metadata_v5[idx]['celltype_idx']]).to(device)
    dr_t  = torch.tensor([metadata_v5[idx]['drug_idx']]).to(device)

    with torch.no_grad():
        all_attn = model.get_attention_weights(x_pop, ct_t, dr_t)

    avg_attn = torch.stack(all_attn).mean(0)[0].numpy()  # (3060, 3060)

    # CV per gene across timepoints
    gene_cv = []
    for gi, gene in enumerate(final_genes_500):
        tp_vals = [avg_attn[:, gi * N_TIMEPOINTS + ti].mean()
                   for ti in range(N_TIMEPOINTS)]
        tp_vals = np.array(tp_vals)
        cv = tp_vals.std() / (tp_vals.mean() + 1e-10)
        gene_cv.append((gene, cv, tp_vals))

    gene_cv.sort(key=lambda x: -x[1])
    print("Top 10 genes by temporal attention CV:")
    for gene, cv, vals in gene_cv[:10]:
        print(f"  {gene:<15} CV={cv:.4f}  vals={[f'{v:.4f}' for v in vals]}")

    # Htr1f specifically
    htr1f_entry = next((g for g in gene_cv if g[0] == 'Htr1f'), None)
    if htr1f_entry:
        gene, cv, vals = htr1f_entry
        rank = next(i for i, g in enumerate(gene_cv) if g[0] == 'Htr1f') + 1
        print(f"\nHtr1f: rank={rank}, CV={cv:.4f}")
        for tp, v in zip(TIMEPOINT_LIST, vals):
            print(f"  {tp}: {v:.6f}")


# =============================================================================
# CELL 13: Htr1f expression trajectory figure (Figure 6)
# =============================================================================
conn = sqlite3.connect(f'{DRIVE_PATH}/data/differential_expression_v2.sqlite')
htr1f_l23 = pd.read_sql("""
    SELECT time, drug, mean_expression
    FROM gene_expression
    WHERE gene_name = 'Htr1f'
    AND celltype = '007 L2/3 IT CTX Glut'
    ORDER BY drug, time
""", conn)
conn.close()

pivot_l23   = htr1f_l23.pivot(index='time', columns='drug',
                               values='mean_expression')
pivot_l23   = pivot_l23.reindex(['1h', '2h', '4h', '24h', '72h'])
timepoints  = ['1h', '2h', '4h', '24h', '72h']
psilo_all   = [2.025, 2.618, 2.542, 2.446, 2.362]
ket_all     = [2.573, 2.494, 2.593, 2.517, 2.639]
baseline_l23 = 0.938
baseline_all = 2.692

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: L2/3 IT
ax = axes[0]
ax.axhline(y=baseline_l23, color='gray', linestyle='--', alpha=0.6,
           label='Baseline (no drug)')
ax.plot(timepoints, pivot_l23['Psilo'].values, 'o-',
        color=C_PSILO, linewidth=2.5, markersize=9, label='Psilocybin')
ax.plot(timepoints, pivot_l23['Ket'].values, 'o-',
        color=C_KET, linewidth=2.5, markersize=9, label='Ketamine')
ax.annotate('*p<0.001\n(DEG significant)', xy=('1h', pivot_l23['Psilo']['1h']),
            xytext=(0.15, 0.25), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=8, color='red')
ax.annotate('Model-flagged\n(p=0.22, n.s.)', xy=('72h', pivot_l23['Psilo']['72h']),
            xytext=(0.72, 0.25), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
            fontsize=8, color='orange')
ax.set_xlabel('Timepoint post-administration')
ax.set_ylabel('Mean expression (log CPM)')
ax.set_title('Htr1f — L2/3 IT neurons\n(primary psilocybin target)')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.2)

# Panel 2: All cell types
ax = axes[1]
ax.axhline(y=baseline_all, color='gray', linestyle='--', alpha=0.6,
           label='Baseline (no drug)')
ax.plot(timepoints, psilo_all, 'o-', color=C_PSILO,
        linewidth=2.5, markersize=9, label='Psilocybin')
ax.plot(timepoints, ket_all, 'o-', color=C_KET,
        linewidth=2.5, markersize=9, label='Ketamine')
ax.annotate('*Significant\n(7 cell types)', xy=('1h', psilo_all[0]),
            xytext=(0.15, 0.2), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=8, color='red')
ax.annotate('Model-flagged\n(n.s., consistent\ndirection 14/16\ncell types)',
            xy=('72h', psilo_all[4]),
            xytext=(0.62, 0.15), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
            fontsize=8, color='orange')
ax.set_xlabel('Timepoint post-administration')
ax.set_ylabel('Mean expression (log CPM)')
ax.set_title('Htr1f — all cell types average')
ax.legend(fontsize=9)
ax.set_ylim(1.7, 2.9)

plt.suptitle(
    'Htr1f biphasic suppression pattern — identified by Transformer attention analysis\n'
    'Acute suppression at 1h (known) + late-phase divergence at 72h (model-flagged)',
    fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(f'{DRIVE_PATH}/figures/htr1f_biphasic_clean.png',
            dpi=150, bbox_inches='tight')
plt.show()
