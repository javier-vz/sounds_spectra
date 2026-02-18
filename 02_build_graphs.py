#!/usr/bin/env python3
"""
STEP 2: Build Graph-of-Words from UDHR texts
Solo para familias americanas grandes (≥5 lenguas)
"""

import os
import pickle
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import numpy as np

print("="*70)
print("STEP 2: CONSTRUCCIÓN DE GRAFOS")
print("="*70)

# Paths
FAMILY_CSV = "americas_families.csv"
UDHR_DIR = "udhr"
OUT_DIR = "outputs/pickles"

os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 2  # Usar window=2 (baseline)

# 1. Cargar familias
print("\n[1/4] Cargando familias americanas...")
df_fam = pd.read_csv(FAMILY_CSV)
lang_to_family = dict(zip(df_fam['iso_code'], df_fam['family']))

# Filtrar familias grandes
family_counts = Counter(lang_to_family.values())
large_families = {f for f, c in family_counts.items() if c >= 5}

print(f"   Total lenguas: {len(lang_to_family)}")
print(f"   Familias grandes (≥5): {len(large_families)}")
for fam in sorted(large_families, key=lambda x: family_counts[x], reverse=True):
    print(f"      {str(fam)[:40]:40s}: {family_counts[fam]:3d}")

# Lenguas a procesar
target_langs = [iso for iso, fam in lang_to_family.items() if fam in large_families]
print(f"\n   Lenguas a procesar: {len(target_langs)}")

# 2. Leer UDHR
print("\n[2/4] Leyendo archivos UDHR...")
languages = {}
missing = []

for iso in target_langs:
    path = os.path.join(UDHR_DIR, f"udhr_{iso}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        languages[iso] = lines
    else:
        missing.append(iso)

print(f"   Cargados: {len(languages)}")
if missing:
    print(f"   Faltantes: {len(missing)}")

# 3. Limpiar y tokenizar
print("\n[3/4] Limpiando y tokenizando...")

punct = r'``!"#$%&\¿()*+,-./:;<=>?@[\]_{|}\'\''
table = str.maketrans({k: None for k in punct})

def clean(iso, sentences):
    cleaned = []
    for s in sentences:
        s = s.translate(table)
        tokens = [w.lower().translate(table) for w in s.split() 
                  if w and not w.isdigit()]
        tokens = [w for w in tokens if len(w) > 1]
        if tokens:
            cleaned.append(tokens)
    
    # Skip header lines
    skip = {'zro': 6, 'tca': 7, 'gyr': 9}.get(iso, 5)
    return cleaned[skip:] if len(cleaned) > skip else cleaned

clean_languages = {}
total_tokens = 0

for iso, sents in languages.items():
    cleaned = clean(iso, sents)
    clean_languages[iso] = cleaned
    total_tokens += sum(len(s) for s in cleaned)

print(f"   Procesadas: {len(clean_languages)} lenguas")
print(f"   Total tokens: {total_tokens:,}")

# 4. Construir grafos
print(f"\n[4/4] Construyendo grafos (window={WINDOW_SIZE})...")

def GoW(sentences, window_size=2):
    tokens = [t for s in sentences for t in s]
    cooc = defaultdict(int)
    
    for i, w1 in enumerate(tokens):
        for j in range(i+1, min(i+window_size+1, len(tokens))):
            w2 = tokens[j]
            if w1 != w2:
                edge = (w1, w2) if w1 <= w2 else (w2, w1)
                cooc[edge] += 1
    
    G = nx.Graph()
    for (w1, w2), weight in cooc.items():
        G.add_edge(w1, w2, weight=weight)
    
    return G

graphs = {}
stats = []

for iso, sents in clean_languages.items():
    G = GoW(sents, WINDOW_SIZE)
    graphs[iso] = G
    
    stats.append({
        'iso': iso,
        'family': lang_to_family[iso],
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
    })

print(f"   Grafos construidos: {len(graphs)}")

# Guardar
with open(os.path.join(OUT_DIR, "graphs.pkl"), "wb") as f:
    pickle.dump(graphs, f)

with open(os.path.join(OUT_DIR, "clean_languages.pkl"), "wb") as f:
    pickle.dump(clean_languages, f)

with open(os.path.join(OUT_DIR, "languages.pkl"), "wb") as f:
    pickle.dump(languages, f)

pd.DataFrame(stats).to_csv(os.path.join(OUT_DIR, "graph_stats.csv"), index=False)

# Stats
df_stats = pd.DataFrame(stats)
print(f"\n   Estadísticas:")
print(f"      Nodos (mean): {df_stats['nodes'].mean():.0f}")
print(f"      Edges (mean): {df_stats['edges'].mean():.0f}")
print(f"      Density (mean): {df_stats['density'].mean():.4f}")

print("\n" + "="*70)
print("✅ PASO 2 COMPLETADO")
print(f"   Archivos en: {OUT_DIR}/")
print("   Siguiente: python 03_compute_spectra.py")
print("="*70)
