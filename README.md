# Spectral Analysis of Language Families via Graph-of-Words Sonification

Pipeline completo para analizar familias lingüísticas americanas usando espectros Laplacianos de grafos de co-ocurrencia léxica.

## 📋 Requisitos

### Archivos de entrada
```
tu_proyecto/
├── udhr/                          # Textos UDHR (udhr_xxx.txt)
├── languoid.csv                   # Glottolog languoid data
└── languages_and_dialects_geo.csv # Macroareas geográficas
```

### Dependencias Python
```bash
pip install numpy pandas networkx scipy scikit-learn matplotlib seaborn
```

## 🚀 Ejecución Completa

### Orden de ejecución (6 pasos):

```bash
# 1. Preparar datos de familias americanas
python 01_prepare_data.py
# Output: americas_families.csv (44 lenguas, 5 familias grandes)

# 2. Construir grafos de co-ocurrencia
python 02_build_graphs.py
# Output: outputs/pickles/ (graphs.pkl, clean_languages.pkl)

# 3. Calcular espectros Laplacianos
python 03_compute_spectra.py
# Output: outputs/spectra/ (eigenvalues, heat traces)

# 4. Sintetizar audios (3 variantes optimizadas)
python 04_synthesize_sounds.py
# Output: outputs/audio/ (V7_sparse, V3_sqrt, V1_linear)

# 5. Análisis estadístico completo
python 05_analyze_spectra.py
# Output: outputs/analysis/ (métricas, distancias, clasificación)

# 6. Generar figuras para paper
python 06_visualize_results.py
# Output: outputs/figures/ (PNG publication-ready)
```

## 📊 Resultados Principales

### Dataset
- **44 lenguas** de las Américas
- **5 familias** grandes (≥5 lenguas):
  - Quechuan (13)
  - Arawakan (8)
  - Otomanguean (8)
  - Mayan (8)
  - Panoan (7)

### Métricas de Discriminación

| Método | Silhouette | Ratio inter/intra | Accuracy |
|--------|------------|-------------------|----------|
| **Espectral directo** | +0.14 | 2.44× | — |
| **V7_sparse (32 bins)** | +0.062 | 2.00× | 57% |
| **V3_sqrt** | +0.044 | 1.87× | 57% |
| **V1_linear (baseline)** | +0.020 | 1.80× | 61% |

### Hallazgos Clave

1. **Control regional es esencial**: Américas (Sil=+0.14) vs Global (Sil=-0.68)
2. **Sparse > Dense**: 32 bins mejor que 64 o 128 (regularización)
3. **Mapeo sqrt óptimo**: Balance entre información y robustez
4. **Sonificación pierde ~70%** de info vs análisis espectral directo

## 📁 Estructura de Salida

```
outputs/
├── pickles/
│   ├── graphs.pkl              # Grafos GoW por lengua
│   ├── clean_languages.pkl     # Tokens limpiados
│   └── graph_stats.csv         # Estadísticas de grafos
│
├── spectra/
│   ├── {lang}_eigvals.npy      # Eigenvalues del Laplaciano
│   └── {lang}_heattrace.npz    # Heat trace Z(t)
│
├── audio/
│   ├── V7_sparse/              # 32 bins (mejor clustering)
│   ├── V3_sqrt/                # sqrt mapping (balance)
│   └── V1_linear/              # 64 bins (baseline)
│
├── analysis/
│   ├── distance_matrix.npy     # Distancias Wasserstein
│   ├── mds_coordinates.npy     # Proyección MDS
│   ├── tsne_coordinates.npy    # Proyección t-SNE
│   ├── classification_results.pkl
│   ├── metadata.pkl
│   └── statistics.txt
│
└── figures/
    ├── mds_clustering.png      # Visualización MDS por familia
    ├── distance_distributions.png
    ├── classification_report.png
    └── summary_comparison.png
```

## 🔬 Para el Paper (Physical Review E)

### Figuras principales recomendadas:
1. **MDS clustering** (outputs/figures/mds_clustering.png)
2. **Distance distributions** (INTRA vs INTER familia)
3. **Silhouette comparison** (Espectral vs Acústico)
4. **Classification accuracy** por familia

### Abstract sketch:
```
We analyze Graph-of-Words Laplacian spectra for 44 American indigenous 
languages from 5 families. While global clustering shows poor family 
discrimination (Silhouette=-0.68), regional control (Americas only) 
reveals significant separation (Silhouette=+0.14, inter/intra=2.44×). 

Sonification via sparse spectral binning (32 bins) achieves 
Silhouette=+0.062 with 211% improvement over baseline, though with 
~70% information loss relative to direct spectral analysis. This 
demonstrates that spectral signatures capture family-specific 
co-occurrence patterns at regional scales.
```

## ⚙️ Parámetros Configurables

### En `02_build_graphs.py`:
- `WINDOW_SIZE = 2` - Ventana de co-ocurrencia (2 = bigramas)

### En `04_synthesize_sounds.py`:
- `SR = 44100` - Sample rate
- `DUR = 10.0` - Duración en segundos
- `VARIANTS` - Agregar/quitar variantes de sonificación

### En `05_analyze_spectra.py`:
- `N_MFCC = 13` - Número de coeficientes MFCC
- `CV_FOLDS = 5` - Folds para cross-validation

## 🐛 Troubleshooting

### Error: "No such file or directory: udhr/"
- Asegúrate de tener la carpeta `udhr/` con archivos `udhr_xxx.txt`

### Error: "No module named 'networkx'"
- Instala: `pip install networkx scipy scikit-learn`

### Warning: "Some families have <2 samples"
- Normal si algunas familias tienen pocas lenguas
- El análisis se enfoca en familias grandes (≥5)

### Archivos de audio no generados
- Verifica que el paso 3 (espectros) se completó
- Revisa que existen archivos `*_eigvals.npy` en `outputs/spectra/`

## 📚 Referencias

- **Graph-of-Words**: Rousseau & Vazirgiannis (2013)
- **Laplacian spectra**: Chung (1997) - Spectral Graph Theory
- **UDHR corpus**: Unicode Common Locale Data Repository
- **Glottolog**: Hammarström et al. (2023)

## 👥 Autores

[Tu nombre aquí]

## 📄 Licencia

[Tu licencia aquí]

---

**Última actualización**: Febrero 2026  
**Para preguntas**: [tu email]
