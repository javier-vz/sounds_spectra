#!/usr/bin/env python3
"""
STEP 4: Synthesize Audio from Spectra
Genera 3 variantes de sonificación optimizadas:
- V7_sparse: 32 bins (mejor clustering)
- V3_sqrt: sqrt mapping (balance)
- V1_linear: 64 bins (baseline)
"""

import os
import pickle
import numpy as np
import wave

print("="*70)
print("STEP 4: SÍNTESIS DE AUDIO")
print("="*70)

# Paths
PICKLES_DIR = "outputs/pickles"
SPECTRA_DIR = "outputs/spectra"
AUDIO_DIR = "outputs/audio"

os.makedirs(AUDIO_DIR, exist_ok=True)

# Audio params
SR = 44100
DUR = 10.0
N_SAMPLES = int(SR * DUR)

# Variantes a generar
VARIANTS = {
    'V7_sparse': {'n_bins': 32, 'mapping': 'linear', 'fmin': 110, 'fmax': 3520},
    'V3_sqrt': {'n_bins': 64, 'mapping': 'sqrt', 'fmin': 110, 'fmax': 3520},
    'V1_linear': {'n_bins': 64, 'mapping': 'linear', 'fmin': 110, 'fmax': 3520},
}

# 1. Cargar grafos
print("\n[1/3] Cargando datos...")
with open(os.path.join(PICKLES_DIR, "graphs.pkl"), "rb") as f:
    graphs = pickle.load(f)

print(f"   Grafos: {len(graphs)}")

# 2. Helper functions
def save_wav(path, x, sr=SR):
    """Guardar WAV mono PCM16"""
    x = np.clip(np.nan_to_num(x), -1.0, 1.0)
    x_int16 = (x * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x_int16.tobytes())

def map_eigenvalues_to_freqs(centers, fmin, fmax, mapping):
    """Mapea eigenvalues → frecuencias"""
    if mapping == 'linear':
        freqs = fmin + (fmax - fmin) * (centers / 2.0)
    elif mapping == 'sqrt':
        freqs = fmin + (fmax - fmin) * np.sqrt(centers / 2.0)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")
    return freqs

def synthesize_timbre(vals, n_bins, fmin, fmax, mapping, duration=DUR, sr=SR):
    """
    Síntesis aditiva desde histograma de eigenvalues.
    """
    # Histogram
    edges = np.linspace(0, 2, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(vals, bins=edges)
    
    # Densidad de probabilidad
    p = (hist + 1e-12) / (hist + 1e-12).sum()
    
    # Frecuencias
    freqs = map_eigenvalues_to_freqs(centers, fmin, fmax, mapping)
    
    # Amplitudes (de-emphasis)
    amp = p / np.sqrt(centers + 1e-3)
    amp = amp / (amp.max() + 1e-12) * 0.9
    
    # Síntesis
    t = np.arange(int(sr * duration)) / sr
    y = sum(a * np.sin(2 * np.pi * f * t) for a, f in zip(amp, freqs))
    
    # Fade in/out
    fade = int(0.05 * sr)
    window = np.ones_like(y)
    window[:fade] = np.linspace(0, 1, fade)
    window[-fade:] = np.linspace(1, 0, fade)
    y *= window
    
    # Normalize
    y = y / (np.abs(y).max() + 1e-12) * 0.98
    
    return y

# 3. Sintetizar
print("\n[2/3] Sintetizando audios...")

for variant_id, params in VARIANTS.items():
    print(f"\n   {variant_id}: {params['n_bins']} bins, {params['mapping']} mapping")
    
    variant_dir = os.path.join(AUDIO_DIR, variant_id)
    os.makedirs(variant_dir, exist_ok=True)
    
    n_success = 0
    
    for lang in sorted(graphs.keys()):
        eig_path = os.path.join(SPECTRA_DIR, f"{lang}_eigvals.npy")
        
        if not os.path.exists(eig_path):
            continue
        
        try:
            vals = np.load(eig_path)
            
            audio = synthesize_timbre(vals, 
                                     params['n_bins'],
                                     params['fmin'],
                                     params['fmax'],
                                     params['mapping'])
            
            output_path = os.path.join(variant_dir, f"{lang}.wav")
            save_wav(output_path, audio)
            
            n_success += 1
            
        except Exception as e:
            print(f"      Error en {lang}: {e}")
    
    print(f"      ✅ {n_success} audios generados")

print("\n[3/3] Resumen...")
print(f"   Variantes generadas: {len(VARIANTS)}")
print(f"   Recomendadas:")
print(f"      • V7_sparse - Mejor clustering (Sil = +0.062)")
print(f"      • V3_sqrt   - Mejor balance")
print(f"      • V1_linear - Baseline")

print("\n" + "="*70)
print("✅ PASO 4 COMPLETADO")
print(f"   Archivos en: {AUDIO_DIR}/")
print("   Siguiente: python 05_analyze_spectra.py")
print("="*70)
