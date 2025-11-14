#!/usr/bin/env python3
"""
Evaluación de una canción individual (fuera del dataset DEAM).

"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import logging
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from CNNModel.paso3_training import CNN_DualBranch_Mejorado

# Configuración de espectrograma (igual que paso1_preprocess.py)
SR = 32000
HOP_LENGTH = 320  
N_MELS = 64
N_FFT = 1024
FMIN = 50
FMAX = 14000

# Configuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """Cargar modelo entrenado"""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        config = checkpoint.get('config', {
            'dropout_rate': 0.28,
            'batch_norm_momentum': 0.99,
            'threshold': 0.25
        })
        
        model = CNN_DualBranch_Mejorado(config)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, config
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        return None, None

def extract_spectrogram_from_audio(audio_path, target_length=4252, save_audio_clip=False, output_dir=None):
    """Extraer espectrograma """
    try:
        # Cargar audio completo (mismo método que paso5)
        y, sr = librosa.load(audio_path, sr=SR)
        
        # Normalización exacta como paso5_pruebas_abiertas.py
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        else:
            logger.warning(f"Audio vacío o silencio detectado en {audio_path}")
            return None, None
        
        # Extraer espectrograma mel con parámetros exactos del entrenamiento
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX
        )
        
        # Convertir a dB (sin normalización adicional)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Añadir dimensión de canal para CNN (1, n_mels, frames)
        mel_db_model = np.expand_dims(mel_db, axis=0)
        
        # Verificar longitud total
        total_frames = mel_db_model.shape[-1]
        
        if total_frames < target_length:
            logger.warning(f"Audio muy corto ({total_frames} frames) para {audio_path}, se omite")
            return None, None
        
        # Desde frame 0 como los clips
        start_frame = 0
        
        # Verificar que el segmento cabe en el audio
        if start_frame + target_length > total_frames:
            logger.error(f"Audio muy corto: {total_frames} frames, necesario: {target_length}")
            return None, None
        
        # Extraer segmento desde frame 0
        end_frame = start_frame + target_length
        mel_db_model = mel_db_model[..., start_frame:end_frame]
        
        # Información de debug
        audio_start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=HOP_LENGTH)
        audio_end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=HOP_LENGTH)
        
        logger.info(f"Espectrograma extraído: {mel_db_model.shape}")
        logger.info(f"Segmento: {audio_start_time:.1f}s - {audio_end_time:.1f}s")
        
        # Guardar clip de audio procesado si se solicita
        audio_clip_path = None
        if save_audio_clip and output_dir is not None:
            try:
                import soundfile as sf
                
                # Calcular índices de muestras de audio correspondientes a los frames
                start_sample = librosa.frames_to_samples(start_frame, hop_length=HOP_LENGTH)
                end_sample = librosa.frames_to_samples(end_frame, hop_length=HOP_LENGTH)
                
                # Extraer el segmento exacto de audio
                audio_segment = y[start_sample:end_sample]
                
                # Crear nombre de archivo para el clip
                song_name = os.path.splitext(os.path.basename(audio_path))[0]
                clip_filename = f"clip_procesado_{song_name.replace(' ', '_')}.wav"
                audio_clip_path = os.path.join(output_dir, clip_filename)
                
                # Guardar el clip en formato WAV
                sf.write(audio_clip_path, audio_segment, SR)
                
                logger.info(f"Clip guardado: {audio_clip_path}")
                logger.info(f"Duración del clip: {len(audio_segment)/SR:.1f}s")
                
            except ImportError:
                logger.warning("soundfile no está instalado. No se puede guardar el clip de audio.")
                logger.info("Para instalar: pip install soundfile")
                audio_clip_path = None
            except Exception as e:
                logger.error(f"Error guardando clip de audio: {str(e)}")
                audio_clip_path = None
        
        return mel_db_model, audio_clip_path
        
    except Exception as e:
        logger.error(f"Error procesando {audio_path}: {str(e)}")
        return None, None

def predict_single_song(model, mel_spec):
    """Realizar predicción """
    if mel_spec is None:
        return None
    
    # Convertir a tensor con la forma exacta que espera el modelo
    # mel_spec ya tiene forma (1, n_mels, frames)
    mel_tensor = torch.FloatTensor(mel_spec).to(device)
    
    # El modelo espera batch dimension, así que usamos unsqueeze(0) si es necesario
    if len(mel_tensor.shape) == 3:
        mel_tensor = mel_tensor.unsqueeze(0)  # (batch_size, channels, n_mels, frames)
    
    logger.debug(f"Tensor shape: {mel_tensor.shape}")
    
    # Predicción
    model.eval()
    with torch.no_grad():
        valence_pred, arousal_pred = model(mel_tensor)
        
    return {
        'valence': float(valence_pred.cpu().numpy()[0]),
        'arousal': float(arousal_pred.cpu().numpy()[0])
    }

def classify_quadrant(valence, arousal, threshold=5.0):
    """Clasificar emoción en cuadrante"""
    if valence >= threshold and arousal >= threshold:
        return "C1: Alto Valence + Alto Arousal (Alegría/Euforia)"
    elif valence < threshold and arousal >= threshold:
        return "C2: Bajo Valence + Alto Arousal (Agitación/Tensión)"
    elif valence < threshold and arousal < threshold:
        return "C3: Bajo Valence + Bajo Arousal (Tristeza/Calma)"
    else:  # valence >= threshold and arousal < threshold
        return "C4: Alto Valence + Bajo Arousal (Paz/Serenidad)"

def find_audio_file(song_name, canciones_dir):
    """Buscar archivo de audio en el directorio con prioridad para coincidencias exactas"""
    if not os.path.exists(canciones_dir):
        logger.error(f"Directorio no encontrado: {canciones_dir}")
        return None
    
    audio_files = [f for f in os.listdir(canciones_dir) 
                  if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]
    
    # Paso 1: Buscar coincidencia exacta (nombre de canción + cualquier extensión soportada)
    supported_extensions = ['.mp3', '.wav', '.flac', '.m4a']
    
    for ext in supported_extensions:
        exact_match = f"{song_name}{ext}"
        for filename in audio_files:
            if filename.lower() == exact_match.lower():
                logger.info(f"Coincidencia exacta encontrada: {filename}")
                return os.path.join(canciones_dir, filename)
    
    # Paso 2: Buscar coincidencia sin artista (que no contenga " - ")
    for filename in audio_files:
        if song_name.lower() in filename.lower() and " - " not in filename:
            logger.info(f"Coincidencia sin artista encontrada: {filename}")
            return os.path.join(canciones_dir, filename)
    
    # Paso 3: Buscar cualquier coincidencia parcial (como fallback)
    for filename in audio_files:
        if song_name.lower() in filename.lower():
            logger.info(f"Coincidencia parcial encontrada: {filename}")
            return os.path.join(canciones_dir, filename)
    
    # No se encontró el archivo
    logger.error(f"No se encontró la canción '{song_name}' en {canciones_dir}")
    return None

def create_visualization(song_name, prediction, output_dir):
    """Crear visualización de la predicción"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Configurar cuadrantes emocionales
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Etiquetas de cuadrantes
    ax.text(7.5, 7.5, 'C1: Alegría\nEuforia', ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    ax.text(2.5, 7.5, 'C2: Agitación\nTensión', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax.text(2.5, 2.5, 'C3: Tristeza\nCalma', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    ax.text(7.5, 2.5, 'C4: Paz\nSerenidad', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    # Predicción - Punto verde simple y claro
    ax.scatter(prediction['valence'], prediction['arousal'], 
               s=300, c='green', marker='o', label='Predicción', 
               alpha=0.8, edgecolors='black', linewidth=2)
    
    # Configuración del gráfico
    ax.set_xlim(1, 9)
    ax.set_ylim(1, 9)
    ax.set_xlabel('Valence (1=Negativo, 9=Positivo)', fontsize=12)
    ax.set_ylabel('Arousal (1=Calma, 9=Activación)', fontsize=12)
    ax.set_title(f'Predicción Emocional: {song_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Añadir información de la predicción
    cuadrante = classify_quadrant(prediction['valence'], prediction['arousal'])
    
    info_text = f"""Valores Predichos:
Valence: {prediction['valence']:.2f}
Arousal: {prediction['arousal']:.2f}

Clasificación:
{cuadrante}"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    
    # Guardar
    plot_path = os.path.join(output_dir, f'prediccion_{song_name.replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico guardado: {plot_path}")

def main():
    # Rutas configurables (modificar según estructura del proyecto)
    CANCION = "cancion_ejemplo"  # Nombre del archivo de audio (sin extensión)
    MODELO = "train_results/best_model.pth" # Modelo entrenado
    CANCIONES_DIR = "canciones_usuario"  # Ruta con canciones propias
    
    # Configuración inicial
    timestamp = int(datetime.now().timestamp())
    output_dir = f"eval_cancion_{CANCION.replace(' ', '_')}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Using device: {device}")
    logger.info("EVALUACIÓN DE CANCIÓN INDIVIDUAL")
    logger.info("=" * 60)
    logger.info(f"Canción: {CANCION}")
    logger.info(f"Modelo: {MODELO}")
    logger.info(f"Directorio canciones: {CANCIONES_DIR}")
    logger.info(f"Directorio salida: {output_dir}")
    logger.info("=" * 60)
    
    # Cargar modelo
    logger.info("Cargando modelo...")
    model, config = load_model(MODELO)
    if model is None:
        return
    
    logger.info("Modelo cargado exitosamente")
    
    # Buscar archivo de audio
    audio_path = find_audio_file(CANCION, CANCIONES_DIR)
    if audio_path is None:
        return
    
    logger.info(f"Archivo encontrado: {os.path.basename(audio_path)}")
    
    # Procesar audio
    logger.info("Procesando audio...")
    mel_spec, audio_clip_path = extract_spectrogram_from_audio(
        audio_path, 
        save_audio_clip=True,
        output_dir=output_dir
    )
    if mel_spec is None:
        logger.error("No se pudo procesar el audio")
        return
    
    # Realizar predicción
    logger.info("Realizando predicción...")
    prediction = predict_single_song(model, mel_spec)
    
    # Clasificar cuadrante
    cuadrante = classify_quadrant(prediction['valence'], prediction['arousal'])
    
    # Mostrar resultados
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS:")
    logger.info(f"   Valence: {prediction['valence']:.3f}")
    logger.info(f"   Arousal: {prediction['arousal']:.3f}")
    logger.info(f"   Cuadrante: {cuadrante}")
    
    # Crear visualización
    logger.info("Creando visualización...")
    create_visualization(CANCION, prediction, output_dir)
    
    # Guardar resultados en archivo
    results_file = os.path.join(output_dir, 'resultados.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"EVALUACIÓN DE CANCIÓN INDIVIDUAL\n")
        f.write(f"================================\n\n")
        f.write(f"Canción: {CANCION}\n")
        f.write(f"Archivo: {os.path.basename(audio_path)}\n")
        f.write(f"Modelo: {MODELO}\n")
        if audio_clip_path:
            f.write(f"Clip procesado: {os.path.basename(audio_clip_path)}\n")
        f.write(f"\nPREDICCIÓN:\n")
        f.write(f"Valence: {prediction['valence']:.3f}\n")
        f.write(f"Arousal: {prediction['arousal']:.3f}\n")
        f.write(f"Cuadrante: {cuadrante}\n\n")
        f.write(f"CONFIGURACIÓN DEL MODELO:\n")
        f.write(f"Dropout rate: {config.get('dropout_rate', 'N/A')}\n")
        f.write(f"Batch norm momentum: {config.get('batch_norm_momentum', 'N/A')}\n")
        f.write(f"Threshold métrica híbrida: {config.get('threshold', 'N/A')}\n")
    
    logger.info(f"Resultados guardados: {results_file}")
    if audio_clip_path:
        logger.info(f"Clip de audio guardado: {audio_clip_path}")
    logger.info("Evaluación completada")

if __name__ == "__main__":
    main()
