#!/usr/bin/env python3
"""
STEP 6: Visualize Results
Genera figuras publication-ready para el paper
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

print("="*70)
print("STEP 6: VISUALIZACIÓN DE RESULTADOS")
print("="*70)

# Paths
FAMILY_CSV = "americas_families.csv"
ANALYSIS_DIR = "outputs/analysis"
FIGURES_DIR = "outputs/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Cargar datos
print("\n[1/5] Cargando datos de análisis...")

with open(os.path.join(ANALYSIS_DIR, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

with open(os.path.join(ANALYSIS_DIR, "classification_results.pkl"), "rb") as f:
    results = pickle.load(f)

D = np.load(os.path.join(ANALYSIS_DIR, "distance_matrix.npy"))
coords_mds = np.load(os.path.join(ANALYSIS_DIR, "mds_coordinates.npy"))
coords_tsne = np.load(os.path.join(ANALYSIS_DIR, "tsne_coordinates.npy"))

languages = metadata['languages']
families = metadata['families']
large_families = metadata['large_families']

print(f"   Lenguas: {len(languages)}")
print(f"   Familias: {len(large_families)}")

# 2. Color palette
print("\n[2/5] Preparando paleta de colores...")

family_colors = {
    fam: color for fam, color in zip(
        sorted(large_families),
        sns.color_palette("Set2", len(large_families))
    )
}

colors = [family_colors[fam] for fam in families]

# 3. FIGURA 1: MDS Clustering
print("\n[3/5] Generando figura MDS...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for fam in sorted(large_families):
    mask = [f == fam for f in families]
    ax.scatter(coords_mds[mask, 0], coords_mds[mask, 1],
              c=[family_colors[fam]], label=str(fam)[:30],
              s=100, alpha=0.7, edgecolors='black', linewidth=1)

ax.set_xlabel('MDS 1', fontsize=12, fontweight='bold')
ax.set_ylabel('MDS 2', fontsize=12, fontweight='bold')
ax.set_title(f'MDS Projection of {len(languages)} American Languages\n'
            f'Silhouette = {results["silhouette"]:+.3f}, Ratio = {results["ratio_inter_intra"]:.2f}×',
            fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='best', fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "mds_clustering.png"), dpi=300, bbox_inches='tight')
print(f"   ✅ Guardado: mds_clustering.png")
plt.close()

# 4. FIGURA 2: t-SNE Clustering
print("\n[4/5] Generando figura t-SNE...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for fam in sorted(large_families):
    mask = [f == fam for f in families]
    ax.scatter(coords_tsne[mask, 0], coords_tsne[mask, 1],
              c=[family_colors[fam]], label=str(fam)[:30],
              s=100, alpha=0.7, edgecolors='black', linewidth=1)

ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
ax.set_title(f't-SNE Projection of {len(languages)} American Languages',
            fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='best', fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "tsne_clustering.png"), dpi=300, bbox_inches='tight')
print(f"   ✅ Guardado: tsne_clustering.png")
plt.close()

# 5. FIGURA 3: Distance distributions
print("\n[5/5] Generando figura de distribuciones...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Histograma INTRA vs INTER
ax = axes[0, 0]
intra = []
inter = []
n = len(families)

for i in range(n):
    for j in range(i+1, n):
        if families[i] == families[j]:
            intra.append(D[i, j])
        else:
            inter.append(D[i, j])

ax.hist(intra, bins=30, alpha=0.7, color='blue', label='INTRA-family', edgecolor='black')
ax.hist(inter, bins=30, alpha=0.7, color='red', label='INTER-family', edgecolor='black')
ax.axvline(np.mean(intra), color='blue', linestyle='--', linewidth=2)
ax.axvline(np.mean(inter), color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Wasserstein Distance', fontsize=11, fontweight='bold')
ax.set_ylabel('Count', fontsize=11, fontweight='bold')
ax.set_title('Distance Distributions', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel B: Boxplot por familia
ax = axes[0, 1]
intra_by_family = []
fam_labels = []

for fam in sorted(large_families):
    fam_indices = [i for i, f in enumerate(families) if f == fam]
    if len(fam_indices) < 2:
        continue
    
    fam_dists = [D[i, j] for i in fam_indices for j in fam_indices if i < j]
    if fam_dists:
        intra_by_family.append(fam_dists)
        fam_labels.append(str(fam)[:20])

ax.boxplot(intra_by_family, labels=fam_labels)
ax.set_ylabel('INTRA-family Distance', fontsize=11, fontweight='bold')
ax.set_title('Intra-family Variability', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

# Panel C: MDS
ax = axes[1, 0]
for fam in sorted(large_families):
    mask = [f == fam for f in families]
    ax.scatter(coords_mds[mask, 0], coords_mds[mask, 1],
              c=[family_colors[fam]], label=str(fam)[:20],
              s=80, alpha=0.7, edgecolors='black', linewidth=1)
ax.set_xlabel('MDS 1', fontsize=11, fontweight='bold')
ax.set_ylabel('MDS 2', fontsize=11, fontweight='bold')
ax.set_title('MDS Projection', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(alpha=0.3)

# Panel D: Metrics summary
ax = axes[1, 1]
ax.axis('off')

metrics_data = [
    ['Metric', 'Value'],
    ['', ''],
    ['Silhouette Score', f"{results['silhouette']:+.4f}"],
    ['Calinski-Harabasz', f"{results['calinski_harabasz']:.2f}"],
    ['Davies-Bouldin', f"{results['davies_bouldin']:.2f}"],
    ['', ''],
    ['Ratio inter/intra', f"{results['ratio_inter_intra']:.3f}×"],
    ['INTRA distance', f"{results['intra_mean']:.4f} ± {results['intra_std']:.4f}"],
    ['INTER distance', f"{results['inter_mean']:.4f} ± {results['inter_std']:.4f}"],
    ['', ''],
    ['RF Accuracy', f"{results['rf_accuracy_mean']:.3f} ± {results['rf_accuracy_std']:.3f}"],
    ['SVM Accuracy', f"{results['svm_accuracy_mean']:.3f} ± {results['svm_accuracy_std']:.3f}"],
]

table = ax.table(cellText=metrics_data, loc='center', cellLoc='left',
                colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Header
for i in range(2):
    table[(0, i)].set_facecolor('#2ca02c')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "summary_4panel.png"), dpi=300, bbox_inches='tight')
print(f"   ✅ Guardado: summary_4panel.png")
plt.close()

# 6. FIGURA EXTRA: Heatmap de distancias
print("\n   Generando heatmap de distancias...")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Ordenar por familia
sorted_indices = sorted(range(len(families)), key=lambda i: (families[i], languages[i]))
D_sorted = D[np.ix_(sorted_indices, sorted_indices)]

im = ax.imshow(D_sorted, cmap='viridis', aspect='auto')
ax.set_title('Wasserstein Distance Matrix (sorted by family)',
            fontsize=13, fontweight='bold', pad=15)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Wasserstein Distance', fontsize=11, fontweight='bold')

# Family boundaries
family_changes = []
current_fam = families[sorted_indices[0]]
for i, idx in enumerate(sorted_indices):
    if families[idx] != current_fam:
        family_changes.append(i)
        current_fam = families[idx]

for pos in family_changes:
    ax.axhline(pos - 0.5, color='red', linewidth=2, alpha=0.7)
    ax.axvline(pos - 0.5, color='red', linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "distance_heatmap.png"), dpi=300, bbox_inches='tight')
print(f"   ✅ Guardado: distance_heatmap.png")
plt.close()

print("\n" + "="*70)
print("✅ PASO 6 COMPLETADO")
print(f"   Figuras guardadas en: {FIGURES_DIR}/")
print("\n   Figuras generadas:")
print("   • mds_clustering.png      - Proyección MDS por familia")
print("   • tsne_clustering.png     - Proyección t-SNE")
print("   • summary_4panel.png      - Panel resumen completo")
print("   • distance_heatmap.png    - Matriz de distancias")
print("="*70)
print("\n🎉 PIPELINE COMPLETO - LISTO PARA EL PAPER!")
print("="*70)
