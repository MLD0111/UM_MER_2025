#!/usr/bin/env python3
"""
Evaluación de un clip individual del test set DEAM.
Permite evaluar un clip específico con métrica híbrida
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import argparse
import logging
from datetime import datetime

# Importar modelo y métrica híbrida
from CNNModel.paso3_training import CNN_DualBranch_Mejorado
from EmotionDetection.Metric.evaluation_metric import evaluate_colors_hybrid_improved

# Configuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_mel_spectrogram(audio_path, target_length=4252):
    """
    Extrae espectrograma mel de un archivo de audio usando las mismas configuraciones que paso1_preprocess.py
    """
    try:
        # Configuración de espectrograma (igual que en paso1_preprocess.py)
        SR = 32000
        HOP_LENGTH = 320
        N_MELS = 64
        N_FFT = 1024
        FMIN = 50
        FMAX = 14000
        
        # Cargar audio
        logger.info(f"   Cargando audio: {os.path.basename(audio_path)}")
        y, _ = librosa.load(audio_path, sr=SR)
        
        # Normalización de amplitud (igual que en paso1)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        else:
            raise ValueError("Audio vacío o silencio detectado")
        
        # Calcular espectrograma mel (exactamente igual que en paso1)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Verificar longitud mínima
        frames = mel_db.shape[-1]
        if frames < target_length:
            raise ValueError(f"Espectrograma muy corto ({frames} frames)")
        
        # Añadir dimensión de canal como en paso1 (Shape: (1, n_mels, frames))
        mel_db_model = np.expand_dims(mel_db, axis=0)
        
        # Verificar longitud mínima
        frames = mel_db_model.shape[-1]
        if frames < target_length:
            raise ValueError(f"Espectrograma muy corto ({frames} frames)")
        
        # Truncar a T_FIXED como en paso1_preprocess.py (tomar primeros frames)
        mel_db_model = mel_db_model[..., :target_length]
        
        logger.info(f"   Espectrograma generado: {mel_db_model.shape}")
        return mel_db_model  # Retorna con dimensión de canal
        
    except Exception as e:
        logger.error(f"   Error procesando audio {audio_path}: {str(e)}")
        return None

def classify_quadrant(valence, arousal, threshold=5.0):
    """Clasifica las predicciones en cuadrantes emocionales"""
    if valence >= threshold and arousal >= threshold:
        return "C1: Alto Valence + Alto Arousal (Alegría/Euforia)"
    elif valence < threshold and arousal >= threshold:
        return "C2: Bajo Valence + Alto Arousal (Agitación/Tensión)"
    elif valence < threshold and arousal < threshold:
        return "C3: Bajo Valence + Bajo Arousal (Tristeza/Calma)"
    else:  # valence >= threshold and arousal < threshold
        return "C4: Alto Valence + Bajo Arousal (Calma/Relajación)"

def predict_single_clip(model, spectrogram):
    """Realiza predicción en un espectrograma individual"""
    model.eval()
    with torch.no_grad():
        # El espectrograma ya viene con dimensión de canal (1, n_mels, frames)
        # Solo añadir dimensión batch
        spec_tensor = torch.FloatTensor(spectrogram).unsqueeze(0).to(device)
        
        # Predicción
        valence_pred, arousal_pred = model(spec_tensor)
        
        return {
            'valence': valence_pred.item(),
            'arousal': arousal_pred.item()
        }

def load_ground_truth(clip_id, annotations_file):
    """Carga los valores ground truth para un clip específico"""
    try:
        df = pd.read_csv(annotations_file)
        # Buscar por el archivo .npy correspondiente
        npy_filename = f"{clip_id}.npy"
        row = df[df['spectrogram'] == npy_filename]
        
        if len(row) == 0:
            logger.warning(f"Clip {clip_id} no encontrado en annotations")
            return None
            
        return {
            'valence': float(row.iloc[0]['valence']),
            'arousal': float(row.iloc[0]['arousal']),
            'valence_std': float(row.iloc[0]['valence_std']),
            'arousal_std': float(row.iloc[0]['arousal_std'])
        }
    except Exception as e:
        logger.error(f"Error cargando ground truth: {str(e)}")
        return None

def create_visualization(clip_id, prediction, ground_truth, output_dir):
    """Crea visualización comparativa de predicción vs ground truth"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cuadrantes emocionales
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    
    # Ground truth
    if ground_truth:
        ax1.scatter(ground_truth['valence'], ground_truth['arousal'], 
                   s=200, c='green', marker='s', label='Ground Truth', alpha=0.8)
        # Error bars
        ax1.errorbar(ground_truth['valence'], ground_truth['arousal'],
                    xerr=ground_truth['valence_std'], yerr=ground_truth['arousal_std'],
                    fmt='none', color='green', alpha=0.5, capsize=5)
    
    # Predicción
    ax1.scatter(prediction['valence'], prediction['arousal'], 
               s=200, c='red', marker='o', label='Predicción', alpha=0.8)
    
    # Configuración del gráfico
    ax1.set_xlim(1, 9)
    ax1.set_ylim(1, 9)
    ax1.set_xlabel('Valence')
    ax1.set_ylabel('Arousal')
    ax1.set_title(f'Clip {clip_id} - Predicción vs Ground Truth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Añadir etiquetas de cuadrantes
    ax1.text(7.5, 7.5, 'C1: Alegría\nEuforia', ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.3))
    ax1.text(2.5, 7.5, 'C2: Agitación\nTensión', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
    ax1.text(2.5, 2.5, 'C3: Tristeza\nCalma', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.3))
    ax1.text(7.5, 2.5, 'C4: Calma\nRelajación', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.3))
    
    # Gráfico de barras comparativo
    categories = ['Valence', 'Arousal']
    if ground_truth:
        pred_values = [prediction['valence'], prediction['arousal']]
        true_values = [ground_truth['valence'], ground_truth['arousal']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2.bar(x - width/2, pred_values, width, label='Predicción', color='red', alpha=0.7)
        ax2.bar(x + width/2, true_values, width, label='Ground Truth', color='green', alpha=0.7)
        
        # Error bars para ground truth
        ax2.errorbar(x + width/2, true_values, 
                    yerr=[ground_truth['valence_std'], ground_truth['arousal_std']],
                    fmt='none', color='black', capsize=5)
    else:
        pred_values = [prediction['valence'], prediction['arousal']]
        ax2.bar(categories, pred_values, color='red', alpha=0.7, label='Predicción')
    
    ax2.set_ylabel('Valor')
    ax2.set_title('Comparación de Valores')
    ax2.set_xticks(x if ground_truth else range(len(categories)))
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.set_ylim(1, 9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    output_path = os.path.join(output_dir, f'clip_{clip_id}_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualización guardada: {output_path}")
    return output_path

def calculate_errors(prediction, ground_truth):
    """Calcula errores entre predicción y ground truth"""
    if not ground_truth:
        return None
        
    valence_error = abs(prediction['valence'] - ground_truth['valence'])
    arousal_error = abs(prediction['arousal'] - ground_truth['arousal'])
    
    # Error cuadrático medio
    mse = np.mean([valence_error**2, arousal_error**2])
    rmse = np.sqrt(mse)
    
    # Error absoluto medio
    mae = np.mean([valence_error, arousal_error])
    
    return {
        'valence_error': valence_error,
        'arousal_error': arousal_error,
        'mae': mae,
        'rmse': rmse
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluar un clip individual del test set DEAM')
    
    # Rutas configurables (modificar según estructura del proyecto)
    parser.add_argument('--clip_id', type=str, default='399',
                       help='ID del clip a evaluar')
    parser.add_argument('--model_path', type=str, 
                       default='train_results/best_model_mejorado.pth',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--clips_dir', type=str,
                       default='dataset/DEAM/clips',
                       help='Directorio de clips de audio')
    parser.add_argument('--annotations_file', type=str,
                       default='dataset/DEAM/test_annotations.csv',
                       help='Archivo de anotaciones del test set')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directorio para resultados')
    
    args = parser.parse_args()
    
    # Configurar directorio de salida
    if args.output_dir is None:
        timestamp = int(datetime.now().timestamp())
        args.output_dir = f"eval_clip_{args.clip_id}_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("EVALUACIÓN DE CLIP INDIVIDUAL")
    logger.info("=" * 60)
    logger.info(f"Clip ID: {args.clip_id}")
    logger.info(f"Modelo: {args.model_path}")
    logger.info(f"Directorio clips: {args.clips_dir}")
    logger.info(f"Directorio salida: {args.output_dir}")
    logger.info("=" * 60)
    
    # Verificar archivo de audio
    audio_file = os.path.join(args.clips_dir, f"{args.clip_id}.mp3")
    if not os.path.exists(audio_file):
        logger.error(f"Audio no encontrado: {audio_file}")
        return
    
    # Cargar modelo
    logger.info("Cargando modelo...")
    if not os.path.exists(args.model_path):
        logger.error(f"Modelo no encontrado: {args.model_path}")
        return
        
    try:
        # Cargar checkpoint y extraer configuración
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        
        # Configuración por defecto del modelo
        config_default = {
            'dropout_rate': 0.28,
            'batch_norm_momentum': 0.99,
            'threshold': 0.25,
            'init_bias_valence': 5.0,
            'init_bias_arousal': 5.0
        }
        
        # Usar configuración del checkpoint o la por defecto
        config = checkpoint.get('config', config_default)
        
        logger.info(f"Configuración del modelo:")
        logger.info(f"   Dropout rate: {config.get('dropout_rate', 'N/A')}")
        logger.info(f"   Batch norm momentum: {config.get('batch_norm_momentum', 'N/A')}")
        logger.info(f"   Threshold: {config.get('threshold', 'N/A')}")
        
        # Crear modelo con configuración
        model = CNN_DualBranch_Mejorado(config)
        
        # Cargar pesos del modelo
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Cargado desde 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Cargado desde 'state_dict'")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Cargado desde checkpoint directo")
        
        model.to(device)
        
        # Verificar información adicional del checkpoint
        if 'epoch' in checkpoint:
            logger.info(f"Modelo entrenado hasta época: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            logger.info(f"Mejor pérdida de validación: {checkpoint['best_val_loss']:.4f}")
        if 'train_loss' in checkpoint:
            logger.info(f"Pérdida de entrenamiento: {checkpoint['train_loss']:.4f}")
        
        logger.info("Modelo cargado exitosamente")
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        logger.error("Verifica que el archivo sea un modelo válido entrenado con paso3_training.py")
        return
    
    # Cargar ground truth
    logger.info("Cargando ground truth...")
    ground_truth = load_ground_truth(args.clip_id, args.annotations_file)
    if ground_truth:
        logger.info(f"Ground truth cargado: V={ground_truth['valence']:.2f}, A={ground_truth['arousal']:.2f}")
    else:
        logger.warning("Ground truth no disponible")
    
    # Procesar audio
    logger.info("Procesando audio...")
    spectrogram = extract_mel_spectrogram(audio_file)
    if spectrogram is None:
        logger.error("Error procesando audio")
        return
    
    # Realizar predicción
    logger.info("Realizando predicción...")
    prediction = predict_single_clip(model, spectrogram)
    
    # Clasificar cuadrante (threshold 5.0 = punto medio en escala 1-9)
    classification_threshold = 5.0
    logger.info(f"Threshold para clasificación de cuadrantes: {classification_threshold}")
    
    # Clasificar cuadrante
    cuadrante_pred = classify_quadrant(prediction['valence'], prediction['arousal'], classification_threshold)
    
    # Mostrar resultados
    logger.info("\n" + "=" * 60)
    logger.info("RESULTADOS")
    logger.info("=" * 60)
    logger.info(f"Clip: {args.clip_id}.mp3")
    logger.info(f"Predicción: V={prediction['valence']:.2f}, A={prediction['arousal']:.2f}")
    logger.info(f"Cuadrante predicho: {cuadrante_pred}")
    
    if ground_truth:
        gt_cuadrante = classify_quadrant(ground_truth['valence'], ground_truth['arousal'], classification_threshold)
        logger.info(f"Ground Truth: V={ground_truth['valence']:.2f}, A={ground_truth['arousal']:.2f}")
        logger.info(f"Cuadrante esperado: {gt_cuadrante}")
        
        # Calcular errores
        errors = calculate_errors(prediction, ground_truth)
        if errors:
            logger.info(f"Error Valence: {errors['valence_error']:.3f}")
            logger.info(f"Error Arousal: {errors['arousal_error']:.3f}")
            logger.info(f"MAE: {errors['mae']:.3f}")
            logger.info(f"RMSE: {errors['rmse']:.3f}")
            
            # Determinar si es acierto en cuadrante
            pred_cuad = cuadrante_pred.split(':')[0]
            gt_cuad = gt_cuadrante.split(':')[0]
            es_acierto = pred_cuad == gt_cuad
            logger.info(f"Clasificación cuadrante: {'CORRECTO' if es_acierto else 'INCORRECTO'}")
            
            # Evaluación con métrica híbrida de colores
            academic_threshold = config.get('threshold', 0.25)
            logger.info(f"\nEVALUACIÓN CON MÉTRICA HÍBRIDA (threshold={academic_threshold}):")
            
            color_results = evaluate_colors_hybrid_improved(
                v_pred=np.array([prediction['valence']]),
                a_pred=np.array([prediction['arousal']]),
                v_true=np.array([ground_truth['valence']]),
                a_true=np.array([ground_truth['arousal']]),
                v_std_individual=np.array([ground_truth['valence_std']]),
                a_std_individual=np.array([ground_truth['arousal_std']]),
                threshold=academic_threshold
            )
            
            # Interpretar resultado
            if color_results['green_cases'] > 0:
                academic_result = "VERDE: Cuadrante correcto + alta precisión"
            elif color_results['yellow_cases'] > 0:
                academic_result = "AMARILLO: Cuadrante correcto + precisión parcial"
            elif color_results['orange_cases'] > 0:
                academic_result = "NARANJA: Cuadrante correcto pero impreciso"
            else:
                academic_result = "ROJO: Cuadrante incorrecto"
            
            logger.info(f"Resultado métrica híbrida: {academic_result}")
            logger.info(f"Casos por color: R={color_results['red_cases']} O={color_results['orange_cases']} Y={color_results['yellow_cases']} G={color_results['green_cases']}")
    
    # Crear visualización
    logger.info("\nGenerando visualización...")
    viz_path = create_visualization(args.clip_id, prediction, ground_truth, args.output_dir)
    
    # Guardar resultados en CSV
    results = {
        'clip_id': args.clip_id,
        'valence_pred': prediction['valence'],
        'arousal_pred': prediction['arousal'],
        'cuadrante_pred': cuadrante_pred,
    }
    
    if ground_truth:
        results.update({
            'valence_true': ground_truth['valence'],
            'arousal_true': ground_truth['arousal'],
            'valence_std': ground_truth['valence_std'],
            'arousal_std': ground_truth['arousal_std'],
            'cuadrante_true': gt_cuadrante,
            'valence_error': errors['valence_error'],
            'arousal_error': errors['arousal_error'],
            'mae': errors['mae'],
            'rmse': errors['rmse'],
            'cuadrante_correcto': es_acierto
        })
    
    # Guardar CSV
    results_df = pd.DataFrame([results])
    csv_path = os.path.join(args.output_dir, f'clip_{args.clip_id}_results.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Resultados guardados: {csv_path}")
    
    logger.info("=" * 60)
    logger.info("EVALUACIÓN COMPLETADA")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
