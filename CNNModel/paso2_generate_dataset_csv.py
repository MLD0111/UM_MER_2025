import pandas as pd
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

"""
Genera archivos CSV de entrenamiento y test a partir del dataset DEAM.

Características principales:
- Split a nivel de canción para evitar data leakage
- Las aumentaciones se conservan solo en el conjunto de entrenamiento
- El conjunto de test contiene únicamente espectrogramas originales sin aumentación
"""

# ==================== CONFIGURACIÓN ====================
# Modificar estas rutas según la ubicación del dataset DEAM
BASE_PATH = "REM/dataset/DEAM"
ANNOTATIONS_FILE = os.path.join(BASE_PATH, "static_annotations_averaged_songs_1_2000.csv")  # Archivo de anotaciones DEAM
NPY_FOLDER = os.path.join(BASE_PATH, "mel_numpy")  # Carpeta con espectrogramas generados en paso1
FULL_CSV = os.path.join(BASE_PATH, "spectrogram_annotations.csv")  # Salida: dataset completo
TRAIN_CSV = os.path.join(BASE_PATH, "train_annotations.csv")  # Salida: conjunto de entrenamiento
TEST_CSV = os.path.join(BASE_PATH, "test_annotations.csv")  # Salida: conjunto de test

# Cargar anotaciones y limpiar nombres de columnas
df = pd.read_csv(ANNOTATIONS_FILE)
df.columns = df.columns.str.strip()

# Seleccionar columnas relevantes
df = df[["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"]]
df["song_id"] = df["song_id"].astype(int).astype(str)

# Crear diccionario de anotaciones por canción
annotation_map = {
    row["song_id"]: {
        "valence": row["valence_mean"],
        "arousal": row["arousal_mean"],
        "valence_std": row["valence_std"],
        "arousal_std": row["arousal_std"]
    }
    for _, row in df.iterrows()
}

# Listar archivos de espectrogramas
all_files = [f for f in os.listdir(NPY_FOLDER) if f.endswith(".npy")]

# Agrupar archivos por canción base
grouped_files = defaultdict(list)
for f in all_files:
    base_id = f.replace(".npy", "").split("_")[0]  # Ejemplo: "0137" de "0137_pitch.npy"
    grouped_files[base_id].append(f)

# Dividir canciones en train/test (80/20)
base_ids = list(grouped_files.keys())
train_ids, test_ids = train_test_split(base_ids, test_size=0.2, random_state=42)

# Construir conjunto de entrenamiento con todas las aumentaciones
train_data = []
test_data = []

for song_id in train_ids:
    if song_id in annotation_map:
        for f in grouped_files[song_id]:
            train_data.append([
                f,
                annotation_map[song_id]["valence"],
                annotation_map[song_id]["arousal"],
                annotation_map[song_id]["valence_std"],
                annotation_map[song_id]["arousal_std"]
            ])

# Construir conjunto de test solo con espectrogramas originales
for song_id in test_ids:
    if song_id in annotation_map:
        original_file = f"{song_id}.npy"
        if original_file in grouped_files[song_id]:
            test_data.append([
                original_file,
                annotation_map[song_id]["valence"],
                annotation_map[song_id]["arousal"],
                annotation_map[song_id]["valence_std"],
                annotation_map[song_id]["arousal_std"]
            ])
        else:
            print(f"No se encontró archivo original para test: {original_file}")

# Guardar archivos CSV
full_df = pd.DataFrame(train_data + test_data, columns=[
    'spectrogram', 'valence', 'arousal', 'valence_std', 'arousal_std'
])
full_df.to_csv(FULL_CSV, index=False)
print(f"Guardado CSV completo con {len(full_df)} muestras: {FULL_CSV}")

pd.DataFrame(train_data, columns=full_df.columns).to_csv(TRAIN_CSV, index=False)
pd.DataFrame(test_data, columns=full_df.columns).to_csv(TEST_CSV, index=False)
print(f"Train set: {len(train_data)} muestras")  # 6975 muestras
print(f"Test set: {len(test_data)} muestras (solo originales)")  # 349 muestras