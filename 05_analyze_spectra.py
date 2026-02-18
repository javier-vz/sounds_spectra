#!/usr/bin/env python3
"""
STEP 5: Statistical Analysis of Spectra
Análisis completo: distancias Wasserstein, clustering, clasificación
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import wasserstein_distance
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import MDS, TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

print("="*70)
print("STEP 5: ANÁLISIS ESTADÍSTICO")
print("="*70)

# Paths
FAMILY_CSV = "americas_families.csv"
SPECTRA_DIR = "outputs/spectra"
ANALYSIS_DIR = "outputs/analysis"

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# 1. Cargar datos
print("\n[1/6] Cargando familias y espectros...")
df_fam = pd.read_csv(FAMILY_CSV)
lang_to_family = dict(zip(df_fam['iso_code'], df_fam['family']))

# Cargar espectros
spectra = {}
for lang in lang_to_family.keys():
    eig_path = os.path.join(SPECTRA_DIR, f"{lang}_eigvals.npy")
    if os.path.exists(eig_path):
        spectra[lang] = np.load(eig_path)

print(f"   Espectros cargados: {len(spectra)}")

# Filtrar familias grandes (≥5)
family_counts = Counter([lang_to_family[lang] for lang in spectra.keys()])
large_families = {f for f, c in family_counts.items() if c >= 5}

langs_filtered = [lang for lang in spectra.keys() 
                  if lang_to_family[lang] in large_families]
families_filtered = [lang_to_family[lang] for lang in langs_filtered]

print(f"   Lenguas (familias ≥5): {len(langs_filtered)}")
print(f"   Familias grandes: {len(large_families)}")
for fam in sorted(large_families, key=lambda x: family_counts[x], reverse=True):
    print(f"      {str(fam)[:40]:40s}: {family_counts[fam]:3d}")

# 2. Features espectrales
print("\n[2/6] Extrayendo features espectrales...")

def spectral_features(vals):
    """15 features from eigenvalue distribution"""
    return np.array([
        np.mean(vals),                    # 0: mean
        np.std(vals),                     # 1: std
        np.percentile(vals, 25),          # 2: Q1
        np.median(vals),                  # 3: median
        np.percentile(vals, 75),          # 4: Q3
        vals[0] if len(vals) > 0 else 0,  # 5: lambda_0
        vals[1] if len(vals) > 1 else 0,  # 6: lambda_1
        vals[2] if len(vals) > 2 else 0,  # 7: lambda_2
        vals[3] if len(vals) > 3 else 0,  # 8: lambda_3
        vals[4] if len(vals) > 4 else 0,  # 9: lambda_4
        (vals[1] - vals[0]) if len(vals) > 1 else 0,  # 10: gap_1
        (vals[2] - vals[1]) if len(vals) > 2 else 0,  # 11: gap_2
        len(vals),                        # 12: spectrum size
        np.sum(vals < 0.1) / len(vals),   # 13: fraction < 0.1
        np.sum(vals > 1.5) / len(vals),   # 14: fraction > 1.5
    ])

X_features = np.array([spectral_features(spectra[lang]) for lang in langs_filtered])
print(f"   Feature matrix: {X_features.shape}")

# 3. Matriz de distancias Wasserstein
print("\n[3/6] Calculando distancias Wasserstein...")

n = len(langs_filtered)
D = np.zeros((n, n))

for i in range(n):
    for j in range(i+1, n):
        d = wasserstein_distance(spectra[langs_filtered[i]], 
                                 spectra[langs_filtered[j]])
        D[i, j] = d
        D[j, i] = d
    
    if (i+1) % 10 == 0:
        print(f"   [{i+1:3d}/{n}] Calculadas...")

np.save(os.path.join(ANALYSIS_DIR, "distance_matrix.npy"), D)
print(f"   ✅ Matriz de distancias: {D.shape}")

# 4. Clustering metrics
print("\n[4/6] Métricas de clustering...")

le = LabelEncoder()
y_encoded = le.fit_transform(families_filtered)

sil = silhouette_score(D, y_encoded, metric='precomputed')
ch = calinski_harabasz_score(X_features, y_encoded)
db = davies_bouldin_score(X_features, y_encoded)

print(f"   Silhouette Score:      {sil:+.4f}")
print(f"   Calinski-Harabasz:     {ch:.2f}")
print(f"   Davies-Bouldin:        {db:.2f}")

# Distancias intra vs inter
intra_dists = []
inter_dists = []

for i in range(n):
    for j in range(i+1, n):
        if families_filtered[i] == families_filtered[j]:
            intra_dists.append(D[i, j])
        else:
            inter_dists.append(D[i, j])

intra_dists = np.array(intra_dists)
inter_dists = np.array(inter_dists)

ratio = inter_dists.mean() / intra_dists.mean()

print(f"\n   Distancias INTRA-familia:")
print(f"      Mean: {intra_dists.mean():.4f}")
print(f"      Std:  {intra_dists.std():.4f}")

print(f"   Distancias INTER-familia:")
print(f"      Mean: {inter_dists.mean():.4f}")
print(f"      Std:  {inter_dists.std():.4f}")

print(f"   Ratio inter/intra: {ratio:.3f}×")

# Por familia
print(f"\n   Distancias INTRA por familia:")
for fam in sorted(large_families):
    fam_indices = [i for i, f in enumerate(families_filtered) if f == fam]
    if len(fam_indices) < 2:
        continue
    
    fam_dists = []
    for i in fam_indices:
        for j in fam_indices:
            if i < j:
                fam_dists.append(D[i, j])
    
    if fam_dists:
        print(f"      {str(fam)[:30]:30s}: {np.mean(fam_dists):.4f} ± {np.std(fam_dists):.4f}")

# 5. Dimensionality reduction
print("\n[5/6] Reducción de dimensionalidad...")

# MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords_mds = mds.fit_transform(D)
np.save(os.path.join(ANALYSIS_DIR, "mds_coordinates.npy"), coords_mds)
print(f"   ✅ MDS: {coords_mds.shape}")

# t-SNE
tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42)
coords_tsne = tsne.fit_transform(D)
np.save(os.path.join(ANALYSIS_DIR, "tsne_coordinates.npy"), coords_tsne)
print(f"   ✅ t-SNE: {coords_tsne.shape}")

# 6. Classification
print("\n[6/6] Clasificación supervisada...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores_rf = cross_val_score(rf, X_features, families_filtered, cv=cv, scoring='accuracy')

print(f"   Random Forest:")
print(f"      Accuracy: {scores_rf.mean():.3f} ± {scores_rf.std():.3f}")

# SVM
svm = SVC(kernel='rbf', random_state=42)
scores_svm = cross_val_score(svm, X_features, families_filtered, cv=cv, scoring='accuracy')

print(f"   SVM (RBF):")
print(f"      Accuracy: {scores_svm.mean():.3f} ± {scores_svm.std():.3f}")

# Guardar resultados
results = {
    'silhouette': float(sil),
    'calinski_harabasz': float(ch),
    'davies_bouldin': float(db),
    'ratio_inter_intra': float(ratio),
    'intra_mean': float(intra_dists.mean()),
    'intra_std': float(intra_dists.std()),
    'inter_mean': float(inter_dists.mean()),
    'inter_std': float(inter_dists.std()),
    'rf_accuracy_mean': float(scores_rf.mean()),
    'rf_accuracy_std': float(scores_rf.std()),
    'svm_accuracy_mean': float(scores_svm.mean()),
    'svm_accuracy_std': float(scores_svm.std()),
    'n_languages': len(langs_filtered),
    'n_families': len(large_families),
}

with open(os.path.join(ANALYSIS_DIR, "classification_results.pkl"), "wb") as f:
    pickle.dump(results, f)

# Metadata
metadata = {
    'languages': langs_filtered,
    'families': families_filtered,
    'family_encoder': le,
    'large_families': list(large_families),
}

with open(os.path.join(ANALYSIS_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

# Summary text
with open(os.path.join(ANALYSIS_DIR, "statistics.txt"), "w") as f:
    f.write("SPECTRAL ANALYSIS RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Dataset: {len(langs_filtered)} languages, {len(large_families)} families\n\n")
    f.write(f"CLUSTERING METRICS:\n")
    f.write(f"  Silhouette Score:      {sil:+.4f}\n")
    f.write(f"  Calinski-Harabasz:     {ch:.2f}\n")
    f.write(f"  Davies-Bouldin:        {db:.2f}\n")
    f.write(f"  Ratio inter/intra:     {ratio:.3f}×\n\n")
    f.write(f"DISTANCES:\n")
    f.write(f"  INTRA-family: {intra_dists.mean():.4f} ± {intra_dists.std():.4f}\n")
    f.write(f"  INTER-family: {inter_dists.mean():.4f} ± {inter_dists.std():.4f}\n\n")
    f.write(f"CLASSIFICATION:\n")
    f.write(f"  Random Forest: {scores_rf.mean():.3f} ± {scores_rf.std():.3f}\n")
    f.write(f"  SVM (RBF):     {scores_svm.mean():.3f} ± {scores_svm.std():.3f}\n")

print("\n" + "="*70)
print("✅ PASO 5 COMPLETADO")
print(f"   Archivos en: {ANALYSIS_DIR}/")
print("   Siguiente: python 06_visualize_results.py")
print("="*70)
