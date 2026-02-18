#!/usr/bin/env python3
"""
STEP 3: Compute Laplacian Spectra and Heat Trace
Calcula eigenvalues del Laplaciano normalizado y heat trace Z(t)
"""

import os
import pickle
import warnings
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh

print("="*70)
print("STEP 3: CÁLCULO DE ESPECTROS LAPLACIANOS")
print("="*70)

# Paths
PICKLES_DIR = "outputs/pickles"
SPECTRA_DIR = "outputs/spectra"

os.makedirs(SPECTRA_DIR, exist_ok=True)

GRAPHS_PKL = os.path.join(PICKLES_DIR, "graphs.pkl")

# 1. Cargar grafos
print("\n[1/3] Cargando grafos...")
with open(GRAPHS_PKL, "rb") as f:
    graphs = pickle.load(f)

print(f"   Grafos cargados: {len(graphs)}")

# 2. Funciones espectrales
def normalized_laplacian(G):
    """Laplaciano normalizado L = I - D^(-1/2) A D^(-1/2)"""
    return nx.normalized_laplacian_matrix(G)

def laplacian_spectrum(L, max_full=1200, k_cap=800):
    """
    Calcula eigenvalues del Laplaciano normalizado.
    - Grafos pequeños (n ≤ max_full): espectro completo
    - Grafos grandes: k_cap eigenvalues más pequeños
    """
    n = L.shape[0]
    
    if n <= 1:
        return np.zeros(1, dtype=np.float64)
    
    if n <= max_full:
        # Espectro completo
        Ld = L.toarray()
        vals = np.linalg.eigvalsh(Ld)
        return np.sort(vals)
    
    # Espectro parcial (k más pequeños)
    k = min(max(2, k_cap), n - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals = eigsh(L, k=k, which="SM", return_eigenvectors=False, tol=1e-8)
    
    return np.sort(vals)

def heat_trace(vals, tmin=1e-3, tmax=1e1, M=300):
    """
    Heat trace: Z(t) = Σ exp(-t·λᵢ)
    Returns: t, Z(t), Z_normalized(t)
    """
    vals = np.asarray(vals, dtype=np.float64)
    t = np.logspace(np.log10(tmin), np.log10(tmax), M)
    
    # Z(t) = sum of exponentials
    Z = np.array([np.exp(-ti * vals).sum() for ti in t], dtype=np.float64)
    
    # Normalize to [0, 1]
    Zt = (Z - Z[-1]) / (Z[0] - Z[-1] + 1e-12)
    
    return t, Z, Zt

# 3. Calcular espectros
print("\n[2/3] Calculando espectros Laplacianos...")

n_total = len(graphs)
for k, (lang, G) in enumerate(sorted(graphs.items()), 1):
    if G.number_of_nodes() < 3:
        print(f"   [{k:3d}/{n_total}] {lang}: grafo muy pequeño, omitido")
        continue
    
    try:
        # Laplaciano normalizado
        L = normalized_laplacian(G)
        
        # Eigenvalues
        vals = laplacian_spectrum(L)
        
        # Heat trace
        t, Z, Zt = heat_trace(vals)
        
        # Guardar
        np.save(os.path.join(SPECTRA_DIR, f"{lang}_eigvals.npy"), vals)
        np.savez_compressed(os.path.join(SPECTRA_DIR, f"{lang}_heattrace.npz"),
                           t=t, Z=Z, Zt=Zt)
        
        if k % 10 == 0:
            print(f"   [{k:3d}/{n_total}] Procesados...")
    
    except Exception as e:
        print(f"   [{k:3d}/{n_total}] {lang}: ERROR → {e}")

print(f"\n   ✅ Espectros calculados para {k} lenguas")

# 4. Estadísticas
print("\n[3/3] Estadísticas de espectros...")

spectrum_sizes = []
for lang in graphs.keys():
    eig_path = os.path.join(SPECTRA_DIR, f"{lang}_eigvals.npy")
    if os.path.exists(eig_path):
        vals = np.load(eig_path)
        spectrum_sizes.append(len(vals))

if spectrum_sizes:
    print(f"   Tamaño de espectros (eigenvalues):")
    print(f"      Min:    {min(spectrum_sizes)}")
    print(f"      Max:    {max(spectrum_sizes)}")
    print(f"      Mean:   {np.mean(spectrum_sizes):.0f}")
    print(f"      Median: {np.median(spectrum_sizes):.0f}")

print("\n" + "="*70)
print("✅ PASO 3 COMPLETADO")
print(f"   Archivos en: {SPECTRA_DIR}/")
print("   Siguiente: python 04_synthesize_sounds.py")
print("="*70)
