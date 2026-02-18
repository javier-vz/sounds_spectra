#!/usr/bin/env python3
"""
STEP 1: Prepare Language Family Data
Crea mapeo ISO → Familia desde Glottolog, filtrado para Américas
"""

import pandas as pd
import os

print("="*70)
print("STEP 1: PREPARACIÓN DE DATOS")
print("="*70)

# Input files
LANGUOID_CSV = "languoid.csv"
GEO_CSV = "languages_and_dialects_geo.csv"
OUTPUT_CSV = "americas_families.csv"

# 1. Cargar Glottolog
print("\n[1/3] Cargando Glottolog languoid...")
df_glot = pd.read_csv(LANGUOID_CSV)

# Filtrar solo lenguas
if 'level' in df_glot.columns:
    df_glot = df_glot[df_glot['level'] == 'language'].copy()

# Buscar columnas
iso_col = next((c for c in ['iso639P3code', 'iso_code', 'iso'] if c in df_glot.columns), None)
family_col = next((c for c in ['family_id', 'top_level_family', 'family', 'parent_id'] if c in df_glot.columns), None)

if not iso_col or not family_col:
    raise ValueError(f"Columnas necesarias no encontradas. ISO: {iso_col}, Family: {family_col}")

df_glot = df_glot[df_glot[iso_col].notna()].copy()

print(f"   ISO codes: columna '{iso_col}'")
print(f"   Familias:  columna '{family_col}'")
print(f"   Lenguas:   {len(df_glot)}")

# 2. Filtrar Américas
print("\n[2/3] Filtrando lenguas de las Américas...")
df_geo = pd.read_csv(GEO_CSV)

americas = df_geo[df_geo['macroarea'].isin(['South America', 'North America'])]
americas_isos = set(americas['isocodes'].dropna())

print(f"   Américas en GEO: {len(americas_isos)} códigos ISO")

# Merge
df_americas = df_glot[df_glot[iso_col].isin(americas_isos)].copy()

print(f"   Match con Glottolog: {len(df_americas)} lenguas")

# 3. Crear dataset final
print("\n[3/3] Creando dataset final...")

output_data = {
    'iso_code': df_americas[iso_col],
    'family': df_americas[family_col],
}

if 'name' in df_americas.columns:
    output_data['language_name'] = df_americas['name']

if 'macroarea' in df_americas.columns:
    output_data['macroarea'] = df_americas['macroarea']

df_final = pd.DataFrame(output_data)
df_final = df_final.drop_duplicates(subset=['iso_code'], keep='first')
df_final.to_csv(OUTPUT_CSV, index=False)

# Stats
family_counts = df_final['family'].value_counts()
large_families = family_counts[family_counts >= 5]

print(f"\n✅ Dataset creado: {OUTPUT_CSV}")
print(f"   Total lenguas: {len(df_final)}")
print(f"   Familias: {len(family_counts)}")
print(f"   Familias grandes (≥5): {len(large_families)}")
print(f"\n   Top 10 familias:")
for fam, count in family_counts.head(10).items():
    print(f"      {str(fam)[:40]:40s}: {count:3d}")

print("\n" + "="*70)
print("✅ PASO 1 COMPLETADO")
print("   Siguiente: python 02_build_graphs.py")
print("="*70)
