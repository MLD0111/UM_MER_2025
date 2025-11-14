import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

# ==================== CONFIGURACIÓN ====================
# Modificar estas rutas según la ubicación del dataset DEAM
BASE_PATH = "REM/dataset/DEAM"  # Ruta base al dataset DEAM
ANNOTATIONS_FILE = os.path.join(BASE_PATH, "static_annotations_averaged_songs_1_2000.csv")  # Archivo de anotaciones DEAM
AUDIO_FOLDER = os.path.join(BASE_PATH, "clips")  # Carpeta con archivos .mp3
NPY_FOLDER = os.path.join(BASE_PATH, "mel_numpy")  # Salida: espectrogramas en formato numpy
PNG_FOLDER = os.path.join(BASE_PATH, "mel_spectrograms")  # Salida: visualizaciones PNG

os.makedirs(NPY_FOLDER, exist_ok=True)
os.makedirs(PNG_FOLDER, exist_ok=True)

# Parámetros para extracción de espectrogramas mel
SR = 32000           # Tasa de muestreo en Hz
HOP_LENGTH = 320     # Salto entre ventanas consecutivas
N_MELS = 64          # Número de bandas mel
N_FFT = 1024         # Tamaño de la FFT
FMIN = 50            # Frecuencia mínima en Hz
FMAX = 14000         # Frecuencia máxima en Hz
T_FIXED = 4252       # Frames por espectrograma (aprox. 42.52 segundos)

# Cargar el archivo de anotaciones y filtrar canciones válidas
annotations = pd.read_csv(ANNOTATIONS_FILE)
song_ids_with_annotations = set(annotations["song_id"].astype(str))

def apply_augmentations(y, sr):
    """
    Aplica técnicas de aumento de datos sobre la señal de audio.
    
    Las transformaciones son sutiles para preservar las características emocionales:
    - pitch: desplazamiento tonal de +1 semitono
    - stretch: extensión temporal a 1.05x la velocidad original
    - pitchstretch: combinación de ambas transformaciones
    - noise: adición de ruido gaussiano con factor 0.003
    
    Args:
        y: señal de audio (numpy array)
        sr: tasa de muestreo
        
    Returns:
        dict con las versiones original y aumentadas del audio
    """
    versions = {"original": y}
    try:
        versions["pitch"] = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
        versions["stretch"] = librosa.effects.time_stretch(y, rate=1.05)
        versions["pitchstretch"] = librosa.effects.pitch_shift(
            librosa.effects.time_stretch(y, rate=1.05), sr=sr, n_steps=1
        )
        noise_factor = 0.003
        noise = np.random.randn(len(y))
        versions["noise"] = np.clip(y + noise_factor * noise, -1, 1)
    except Exception as e:
        print(f"Error en augmentación: {e}")
    return versions

print("Procesando clips de audio...")
clips_processed = 0
total_clips = len([f for f in os.listdir(AUDIO_FOLDER) if f.endswith(".mp3")])

for file in tqdm(os.listdir(AUDIO_FOLDER), desc="Procesando clips", total=total_clips):
    if not file.endswith(".mp3"):
        continue

    song_id = file.replace(".mp3", "")
    if song_id not in song_ids_with_annotations:
        continue

    try:
        # Cargar audio y normalizar amplitud
        y, _ = librosa.load(os.path.join(AUDIO_FOLDER, file), sr=SR)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        else:
            tqdm.write(f"Audio vacío o silencio detectado en {file}, se salta")
            continue

        # Generar versiones aumentadas del audio
        versions = apply_augmentations(y, SR)

        for suffix, y_aug in versions.items():
            # Calcular espectrograma mel
            mel = librosa.feature.melspectrogram(
                y=y_aug,
                sr=SR,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                fmin=FMIN,
                fmax=FMAX
            )
            # Convertir a escala de decibelios
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Preparar datos para la CNN agregando dimensión de canal
            filename = f"{song_id}.npy" if suffix == "original" else f"{song_id}_{suffix}.npy"
            mel_db_model = np.expand_dims(mel_db, axis=0)  # Shape: (1, n_mels, frames)
            
            # Verificar longitud mínima y truncar a T_FIXED frames
            frames = mel_db_model.shape[-1]
            if frames < T_FIXED:
                tqdm.write(f"Espectrograma muy corto ({frames} frames) para {filename}, se omite")
                continue
            mel_db_model = mel_db_model[..., :T_FIXED]

            # Guardar representación numpy
            np.save(os.path.join(NPY_FOLDER, filename), mel_db_model)

            # Generar y guardar visualización como PNG
            fig = plt.figure(figsize=(mel_db.shape[1] / 100, 2.56), dpi=100)
            librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH, 
                                    y_axis='mel', x_axis='time', cmap='magma')
            plt.axis('off')
            plt.gca().set_position([0, 0, 1, 1])
            png_filename = filename.replace(".npy", ".png")
            plt.savefig(os.path.join(PNG_FOLDER, png_filename), bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        clips_processed += 1
        if clips_processed % 10 == 0:
            tqdm.write(f"Procesados {clips_processed} clips")

    except Exception as e:
        tqdm.write(f"Error procesando {file}: {e}")

print(f"Procesamiento completado! Total procesado: {clips_processed} clips")
print(f"Resultados guardados en:")
print(f"   - NPY: {NPY_FOLDER}")
print(f"   - PNG: {PNG_FOLDER}")