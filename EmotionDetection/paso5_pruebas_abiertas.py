#!/usr/bin/env python3
"""
Pruebas abiertas con canciones nuevas.

Evalúa el modelo CNN dual-branch en canciones de diferentes cuadrantes emocionales.

IMPORTANTE: Este script NO usa la métrica híbrida porque:
- No hay ground truth anotado por expertos
- No hay desviaciones estándar individuales  
- Solo evalúa coincidencia de cuadrante con el criterio del equipo de trabajo

Métricas utilizadas:
- Clasificación por cuadrantes
- Comparación con expectativa definida por el equipo
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime

# Importar modelo
from CNNModel.paso3_training import CNN_DualBranch_Mejorado

# Configuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== CONFIGURACIÓN DE CANCIONES ====================
# INSTRUCCIONES: Configurar las canciones a evaluar
# Agregar archivos MP3 en el directorio especificado y definir la estructura esperada
# 
# Formato de cada canción:
# {
#   "archivo": "nombre_del_archivo.mp3",  # Nombre exacto del archivo
#   "nombre": "Nombre Canción",            # Nombre descriptivo
#   "artista": "Nombre Artista",           # Artista
#   "emocion_esperada": "descripción"      # Emoción que esperas detectar
# }
#
# Los cuadrantes son:
# - cuadrante_1_alto_valence_alto_arousal: Alegría/Euforia (música energética y positiva)
# - cuadrante_2_bajo_valence_alto_arousal: Agitación/Tensión (música energética pero negativa)
# - cuadrante_3_bajo_valence_bajo_arousal: Tristeza/Calma melancólica (música tranquila y triste)
# - cuadrante_4_alto_valence_bajo_arousal: Paz/Serenidad (música tranquila y positiva)

CANCIONES_TEST = {
    "cuadrante_1_alto_valence_alto_arousal": [
        # Ejemplo: {"archivo": "cancion_alegre.mp3", "nombre": "Canción Alegre", "artista": "Artista", "emocion_esperada": "Alegría energética"}
    ],
    
    "cuadrante_2_bajo_valence_alto_arousal": [
        # Ejemplo: {"archivo": "cancion_intensa.mp3", "nombre": "Canción Intensa", "artista": "Artista", "emocion_esperada": "Agitación"}
    ],
    
    "cuadrante_3_bajo_valence_bajo_arousal": [
        # Ejemplo: {"archivo": "cancion_triste.mp3", "nombre": "Canción Triste", "artista": "Artista", "emocion_esperada": "Tristeza melancólica"}
    ],
    
    "cuadrante_4_alto_valence_bajo_arousal": [
        # Ejemplo: {"archivo": "cancion_relajante.mp3", "nombre": "Canción Relajante", "artista": "Artista", "emocion_esperada": "Paz y serenidad"}
    ]
}
# ==================== FIN CONFIGURACIÓN ====================

def detect_audio_start(y, sr, hop_length, threshold_db=-40, min_duration=0.5):
    """
    Detecta el inicio real del audio saltando silencios iniciales.
    
    Útil para canciones que pueden tener silencios o intros largas.
    
    Args:
        y: señal de audio
        sr: sample rate
        hop_length: hop length para frames
        threshold_db: umbral de detección en dB
        min_duration: duración mínima de audio activo en segundos
    
    Returns:
        start_frame: frame donde empieza el audio activo
    """
    try:
        # Calcular envolvente de potencia
        frame_length = hop_length * 2
        hop = hop_length
        
        # RMS para detectar energía
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        
        # Convertir a dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Encontrar frames que superan el umbral
        active_frames = np.where(rms_db > threshold_db)[0]
        
        if len(active_frames) == 0:
            # Audio muy silencioso, usar umbral más permisivo
            threshold_permissive = threshold_db - 10
            active_frames = np.where(rms_db > threshold_permissive)[0]
            if len(active_frames) == 0:
                logger.warning("Audio muy silencioso, usando inicio absoluto")
                return 0
        
        # Buscar secuencia continua de frames activos
        first_active = active_frames[0]
        min_frames = int(min_duration * sr / hop_length)
        
        # Si la secuencia requerida es muy larga, usar criterio más flexible
        if min_frames > len(active_frames):
            min_frames = max(1, len(active_frames) // 2)
        
        # Verificar que hay suficiente audio activo continuo
        for i in range(len(active_frames) - min_frames + 1):
            consecutive_count = 1
            for j in range(i + 1, min(i + min_frames, len(active_frames))):
                if active_frames[j] - active_frames[j-1] <= 2:  # Permitir pequeños gaps
                    consecutive_count += 1
                else:
                    break
            
            if consecutive_count >= min_frames:
                return active_frames[i]
        
        # Si no encuentra secuencia continua suficiente, usar primer frame activo
        return first_active
        
    except Exception as e:
        logger.warning(f"Error detectando inicio de audio: {e}, usando inicio absoluto")
        return 0
        return 0

def extract_mel_spectrogram(audio_path, target_length=4252):
    """
    Extrae espectrograma mel de un archivo de audio desde el inicio activo.
    
    Compatible con los parámetros usados en el entrenamiento del modelo.
    
    Args:
        audio_path: ruta al archivo de audio
        target_length: longitud objetivo en frames
    """
    try:
        # Parámetros del modelo
        SR = 32000
        HOP_LENGTH = 320
        N_MELS = 64
        N_FFT = 1024
        FMIN = 50
        FMAX = 14000
        
        # Cargar audio
        y, _ = librosa.load(audio_path, sr=SR)
        
        # Normalizar amplitud
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        else:
            logger.warning(f"Audio vacío o silencio detectado en {audio_path}")
            return None
        
        # Extraer espectrograma mel
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX
        )
        
        # Convertir a dB
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Añadir dimensión de canal para CNN
        mel_db_model = np.expand_dims(mel_db, axis=0)
        
        # Verificar longitud total
        total_frames = mel_db_model.shape[-1]
        
        if total_frames < target_length:
            logger.warning(f"Audio muy corto ({total_frames} frames) para {audio_path}, se omite")
            return None
        
        # Detectar inicio activo para evitar silencios
        start_frame = detect_audio_start(y, SR, HOP_LENGTH)
        
        # Verificar que el segmento cabe en el audio
        if start_frame + target_length > total_frames:
            start_frame = max(0, total_frames - target_length)
        
        # Extraer segmento
        end_frame = start_frame + target_length
        mel_db_model = mel_db_model[..., start_frame:end_frame]
        
        return mel_db_model
        
    except Exception as e:
        logger.error(f"Error procesando {audio_path}: {e}")
        return None

def load_model(model_path):
    """Carga modelo entrenado desde archivo"""
    logger.info(f"Cargando modelo desde: {model_path}")
    
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
        logger.error(f"Error cargando modelo: {e}")
        raise

def predict_song_emotion(model, audio_path):
    """Predice la emoción de una canción desde el inicio activo"""
    # Extraer espectrograma
    mel_spec = extract_mel_spectrogram(audio_path)
    if mel_spec is None:
        return None
    
    # Convertir a tensor
    mel_tensor = torch.FloatTensor(mel_spec).to(device)
    
    # Añadir dimensión de batch si es necesario
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
    """Clasifica emoción en cuadrante según valores de valence y arousal"""
    if valence >= threshold and arousal >= threshold:
        return "C1: Alto Valence + Alto Arousal (Alegría/Euforia)"
    elif valence < threshold and arousal >= threshold:
        return "C2: Bajo Valence + Alto Arousal (Agitación/Tensión)"
    elif valence < threshold and arousal < threshold:
        return "C3: Bajo Valence + Bajo Arousal (Tristeza/Calma)"
    else:  # valence >= threshold and arousal < threshold
        return "C4: Alto Valence + Bajo Arousal (Paz/Serenidad)"

def convert_cuadrante_format(cuadrante_pred):
    """Convierte formato de cuadrante predicho a formato esperado para comparación"""
    mapeo = {
        "C1: Alto Valence + Alto Arousal (Alegría/Euforia)": "cuadrante_1_alto_valence_alto_arousal",
        "C2: Bajo Valence + Alto Arousal (Agitación/Tensión)": "cuadrante_2_bajo_valence_alto_arousal", 
        "C3: Bajo Valence + Bajo Arousal (Tristeza/Calma)": "cuadrante_3_bajo_valence_bajo_arousal",
        "C4: Alto Valence + Bajo Arousal (Paz/Serenidad)": "cuadrante_4_alto_valence_bajo_arousal"
    }
    return mapeo.get(cuadrante_pred, cuadrante_pred)

def find_audio_file(cancion_nombre, canciones_dir):
    """Busca archivo de audio en el directorio con mapeo inteligente"""
    # Búsqueda por coincidencia parcial
    for archivo in os.listdir(canciones_dir):
        if archivo.lower().endswith('.mp3'):
            nombre_limpio = cancion_nombre.lower().replace(" ", "").replace("'", "")
            archivo_limpio = archivo.lower().replace(" ", "").replace("-", "").replace("_", "").replace("'", "")
            
            if nombre_limpio in archivo_limpio or any(palabra in archivo_limpio for palabra in nombre_limpio.split() if len(palabra) > 3):
                return os.path.join(canciones_dir, archivo)
    
    return None

def test_canciones_populares(model, canciones_dir, output_dir, config):
    """Evalúa canciones con detección de inicio activo (sin silencios)"""
    logger.info("PRUEBAS ABIERTAS - CNN_DualBranch")
    logger.info("Canciones seleccionadas - Comportamiento real del modelo")
    logger.info("Detección de inicio activo (sin silencios)")
    logger.info("="*80)
    
    results = []
    aciertos_totales = 0
    total_canciones = 0
    resumen_cuadrantes = {}
    
    for cuadrante, canciones in CANCIONES_TEST.items():
        logger.info(f"\nEVALUANDO {cuadrante.upper()}:")
        logger.info("-" * 60)
        
        aciertos_cuadrante = 0
        cuadrante_esperado_short = "C" + cuadrante.split('_')[1]  # C1, C2, C3, C4
        
        for cancion in canciones:
            # Usar archivo si está especificado
            if 'archivo' in cancion:
                audio_file = os.path.join(canciones_dir, cancion['archivo'])
            else:
                # Buscar archivo por nombre
                audio_file = find_audio_file(cancion['nombre'], canciones_dir)
            
            if not os.path.exists(audio_file):
                logger.warning(f"Audio no encontrado: {cancion.get('archivo', cancion['nombre'])}")
                continue
            
            # Predecir emoción usando inicio activo
            prediction = predict_song_emotion(model, audio_file)
            if prediction is None:
                logger.error(f"Error procesando: {cancion['nombre']}")
                continue
            
            # Clasificar cuadrante
            classification_threshold = 5.0
            cuadrante_pred, emoji = classify_quadrant(prediction['valence'], prediction['arousal'], classification_threshold)
            
            # Verificar acierto comparando cuadrantes
            cuadrante_pred_short = cuadrante_pred.split(':')[0]
            es_exito = cuadrante_pred_short == cuadrante_esperado_short
            
            if es_exito:
                aciertos_cuadrante += 1
                aciertos_totales += 1
            
            total_canciones += 1
            
            # Guardar resultado
            result = {
                'archivo': cancion.get('archivo', ''),
                'cancion': cancion['nombre'],
                'artista': cancion['artista'],
                'emocion_esperada': cancion['emocion_esperada'],
                'cuadrante_esperado': cuadrante,
                'cuadrante_esperado_short': cuadrante_esperado_short,
                'valence_pred': prediction['valence'],
                'arousal_pred': prediction['arousal'],
                'cuadrante_pred': cuadrante_pred,
                'cuadrante_pred_short': cuadrante_pred_short,
                'emoji': emoji,
                'es_exito': es_exito
            }
            results.append(result)
            
            # Log resultado
            exito_mark = "[OK]" if es_exito else "[X]"
            if es_exito:
                logger.info(f"   {exito_mark} {cancion['nombre'][:25]:<25} → {cuadrante_pred_short}")
            else:
                logger.info(f"   {exito_mark} {cancion['nombre'][:25]:<25} → {cuadrante_pred_short} (esperado {cuadrante_esperado_short})")
        
        # Resumen del cuadrante
        total_cuadrante = len(canciones)
        precision_cuadrante = (aciertos_cuadrante / total_cuadrante * 100) if total_cuadrante > 0 else 0
        resumen_cuadrantes[cuadrante] = {
            'aciertos': aciertos_cuadrante,
            'total': total_cuadrante,
            'precision': precision_cuadrante
        }
        logger.info(f"   Precisión {cuadrante_esperado_short}: {aciertos_cuadrante}/{total_cuadrante} = {precision_cuadrante:.1f}%")
    
    # Resumen final
    precision_general = (aciertos_totales / total_canciones * 100) if total_canciones > 0 else 0
    
    logger.info("\n" + "="*80)
    logger.info("RESUMEN FINAL")
    logger.info("="*80)
    
    for cuadrante, stats in resumen_cuadrantes.items():
        cuad_num = cuadrante.split('_')[1]
        logger.info(f"C{cuad_num}: {stats['aciertos']}/{stats['total']} = {stats['precision']:.1f}%")
    
    logger.info(f"\nPRECISIÓN GENERAL: {aciertos_totales}/{total_canciones} = {precision_general:.1f}%")
    
    # Mostrar errores
    errores = [r for r in results if not r['es_exito']]
    if errores:
        logger.info(f"\nPREDICCIONES INCORRECTAS ({len(errores)} canciones):")
        for error in errores:
            logger.info(f"   {error['cancion']} → Esperado: {error['cuadrante_esperado_short']}, Obtenido: {error['cuadrante_pred_short']}")
    else:
        logger.info("\nTODAS LAS PREDICCIONES FUERON CORRECTAS")
    
    # Crear visualización
    create_emotion_map(results, output_dir)
    
    # Guardar resultados
    save_results_popular_songs(results, output_dir)
    
    return results

def create_emotion_map(results, output_dir):
    """Crea mapa emocional de las predicciones"""
    plt.figure(figsize=(16, 12))
    
    # Colores por cuadrante
    colores = {'C1': 'red', 'C2': 'orange', 'C3': 'green', 'C4': 'blue'}
    markers = {'C1': 'o', 'C2': 's', 'C3': '^', 'C4': 'D'}
    
    # Plot: verde para aciertos, rojo para errores
    for i, result in enumerate(results):
        es_exito = result['es_exito']
        
        color = 'green' if es_exito else 'red'
        
        plt.scatter(result['valence_pred'], result['arousal_pred'], 
                   c=color, s=120, alpha=0.8, edgecolors='black', linewidth=1.5)
        
        # Etiqueta con canción y artista
        label_text = f"{result['cancion']}\n{result['artista']}"
        plt.annotate(label_text, 
                    (result['valence_pred'], result['arousal_pred']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=7, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='gray'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.6))
    
    # Divisiones de cuadrantes
    plt.axhline(y=5.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    plt.axvline(x=5.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Estadísticas generales
    total_aciertos = sum(1 for r in results if r['es_exito'])
    total_canciones = len(results)
    precision_general = (total_aciertos / total_canciones * 100) if total_canciones > 0 else 0
    
    # Precisión por cuadrante
    stats_cuadrantes = {}
    for result in results:
        cuad = result['cuadrante_esperado_short']
        if cuad not in stats_cuadrantes:
            stats_cuadrantes[cuad] = {'total': 0, 'aciertos': 0}
        stats_cuadrantes[cuad]['total'] += 1
        if result['es_exito']:
            stats_cuadrantes[cuad]['aciertos'] += 1
    
    # Etiquetas de cuadrantes
    plt.text(6.5, 6.5, 'C1 - Alegría/Euforia', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
    
    plt.text(3.5, 6.5, 'C2 - Agitación/Tensión', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='orange', alpha=0.8))
    
    plt.text(3.5, 3.5, 'C3 - Tristeza/Calma', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    plt.text(6.5, 3.5, 'C4 - Paz/Serenidad', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Configuración del gráfico
    plt.xlabel('Valence', fontsize=12, fontweight='bold')
    plt.ylabel('Arousal (Energía/Calma)', fontsize=12, fontweight='bold')
    plt.title(f'Pruebas Abiertas - CNN_DualBranch\nPrecisión: {precision_general:.1f}% ({total_aciertos}/{total_canciones})', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Leyenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Predicción Correcta'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Predicción Incorrecta')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.xlim(2.5, 7.0)
    plt.ylim(2.5, 7.0)
    
    # Guardar
    timestamp = int(datetime.now().timestamp())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'demo_set_final_balanceado_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Mapa emocional guardado en: {output_dir}")
    
    # Crear segundo gráfico más detallado
    plt.figure(figsize=(14, 10))
    
    # Líneas divisorias
    plt.axhline(y=5, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=5, color='black', linestyle='--', alpha=0.5)
    
    # Plot predicciones
    for result in results:
        es_exito = result['es_exito']
        color = 'green' if es_exito else 'red'
        plt.scatter(result['valence_pred'], result['arousal_pred'], 
                   c=color, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Etiquetas de cuadrantes
    plt.text(7.5, 7.5, 'ALEGRÍA/EUFORIA\n(Alto Valence + Alto Arousal)', 
             ha='center', va='center', fontsize=10, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.text(2.5, 7.5, 'AGITACIÓN/TENSIÓN\n(Bajo Valence + Alto Arousal)', 
             ha='center', va='center', fontsize=10, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    plt.text(2.5, 2.5, 'TRISTEZA/CALMA\n(Bajo Valence + Bajo Arousal)', 
             ha='center', va='center', fontsize=10, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.text(7.5, 2.5, 'PAZ/SERENIDAD\n(Alto Valence + Bajo Arousal)', 
             ha='center', va='center', fontsize=10, weight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.xlim(1, 9)
    plt.ylim(1, 9)
    plt.xlabel('Valence (1-9)', fontsize=12)
    plt.ylabel('Arousal (1-9)', fontsize=12)
    plt.title('Mapa Emocional - Predicciones CNN_DualBranch', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Predicción Correcta'),
        Patch(facecolor='red', label='Predicción Incorrecta')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_map_canciones_populares.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def save_results_popular_songs(results, output_dir):
    """Guarda resultados en CSV y resumen en TXT"""
    # CSV detallado
    df_data = []
    for result in results:
        cuadrante_pred_normalizado = convert_cuadrante_format(result['cuadrante_pred'])
        df_data.append({
            'cancion': result['cancion'],
            'artista': result['artista'],
            'valence_predicho': round(result['valence_pred'], 3),
            'arousal_predicho': round(result['arousal_pred'], 3),
            'cuadrante_predicho': result['cuadrante_pred'],
            'emoji_cuadrante': result['emoji'],
            'cuadrante_esperado': result['cuadrante_esperado'],
            'emocion_esperada': result['emocion_esperada'],
            'coincide_cuadrante': cuadrante_pred_normalizado == result['cuadrante_esperado']
        })
    
    df = pd.DataFrame(df_data)
    csv_path = os.path.join(output_dir, 'predicciones_canciones_populares.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"CSV guardado: {csv_path}")
    
    # Resumen en texto
    with open(os.path.join(output_dir, 'resumen_canciones_populares.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PRUEBAS ABIERTAS - CANCIONES\n")
        f.write("="*70 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modelo: CNN_DualBranch\n")
        f.write(f"Total canciones: {len(results)}\n\n")
        
        # Estadísticas
        aciertos = sum(1 for r in df_data if r['coincide_cuadrante'])
        f.write(f"PRECISIÓN:\n")
        f.write(f"- Correctas: {aciertos}/{len(results)} ({aciertos/len(results)*100:.1f}%)\n\n")
        
        # Por cuadrante
        cuadrantes = {}
        for result in results:
            cuad = result['emoji']
            if cuad not in cuadrantes:
                cuadrantes[cuad] = []
            cuadrantes[cuad].append(result)
        
        for emoji, canciones in cuadrantes.items():
            f.write(f"{emoji} CUADRANTE ({len(canciones)} canciones):\n")
            for cancion in canciones:
                cuadrante_pred_normalizado = convert_cuadrante_format(cancion['cuadrante_pred'])
                es_exito = cuadrante_pred_normalizado == cancion['cuadrante_esperado']
                exito_mark = " [OK]" if es_exito else " [ERROR]"
                
                f.write(f"  • {cancion['cancion']} - {cancion['artista']}{exito_mark}\n")
                f.write(f"    Valence: {cancion['valence_pred']:.2f} | Arousal: {cancion['arousal_pred']:.2f}\n")
                f.write(f"    Esperado: {cancion['emocion_esperada']}\n\n")
    
    logger.info(f"� Resumen guardado: {os.path.join(output_dir, 'resumen_canciones_populares.txt')}")

def main():
    """Función principal para pruebas abiertas"""
    # ==================== CONFIGURACIÓN ====================
    # IMPORTANTE: Configurar estas rutas
    
    # Ruta al modelo entrenado (.pth)
    # Ejemplo: "train_results/best_model.pth"
    model_path = "RUTA_AL_MODELO.pth"
    
    # Ruta donde están los archivos MP3
    # Ejemplo: "canciones_prueba"
    canciones_dir = "DIRECTORIO_CANCIONES"
    
    # Ruta de salida para resultados
    output_dir = f"pruebas_abiertas_{int(datetime.now().timestamp())}"
    # =======================================================
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("PRUEBAS ABIERTAS")
    logger.info("="*70)
    logger.info(f"Canciones: {canciones_dir}")
    logger.info(f"Resultados: {output_dir}")
    logger.info("")
    # Contar canciones
    total_canciones = sum(len(canciones) for canciones in CANCIONES_TEST.values())
    logger.info(f"Evaluando {total_canciones} canciones")
    logger.info("="*70)
    
    # Verificar archivos disponibles
    archivos_encontrados = 0
    archivos_faltantes = []
    
    for cuadrante, canciones in CANCIONES_TEST.items():
        for cancion in canciones:
            if 'archivo' in cancion:
                audio_file = os.path.join(canciones_dir, cancion['archivo'])
            else:
                audio_file = find_audio_file(cancion['nombre'], canciones_dir)
            
            if audio_file and os.path.exists(audio_file):
                archivos_encontrados += 1
            else:
                archivos_faltantes.append(cancion['nombre'])
    
    if archivos_encontrados == 0:
        logger.error("No se encontraron archivos de audio. Verifica:")
        logger.error(f"1. Que el directorio '{canciones_dir}' existe")
        logger.error("2. Que los archivos MP3 están en ese directorio")
        logger.error("3. Los nombres en CANCIONES_TEST")
        return
    
    if archivos_faltantes:
        logger.warning(f"Archivos no encontrados ({len(archivos_faltantes)}):")
        for faltante in archivos_faltantes:
            logger.warning(f"   - {faltante}")
    
    logger.info(f"Encontrados {archivos_encontrados}/{total_canciones} archivos")
    
    # Cargar modelo
    model, config = load_model(model_path)
    
    # Configuración
    logger.info(f"Threshold: {config.get('threshold', 'N/A')}")
    
    # Evaluar
    results = test_canciones_populares(model, canciones_dir, output_dir, config)
    
    # Estadísticas finales
    aciertos = 0
    for result in results:
        cuadrante_pred_normalizado = convert_cuadrante_format(result['cuadrante_pred'])
        if cuadrante_pred_normalizado == result['cuadrante_esperado']:
            aciertos += 1
    
    errores = len(results) - aciertos
    porcentaje = (aciertos / len(results)) * 100 if len(results) > 0 else 0
    
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info(f"{len(results)} canciones: {aciertos} correctas, {errores} incorrectas ({porcentaje:.1f}%)")
    logger.info(f"Resultados en: {output_dir}")

if __name__ == "__main__":
    main()
