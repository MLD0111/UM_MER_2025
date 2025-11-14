#!/usr/bin/env python3
"""
Evaluación del modelo CNN dual-branch entrenado.

Genera métricas de evaluación y visualizaciones:
- Métricas estándar (MSE, MAE, R²) para valence y arousal
- Sistema de evaluación por colores (métrica híbrida)
- Visualizaciones de predicciones vs ground truth
- Exportación de resultados en CSV y JSON
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import argparse
import pandas as pd
from datetime import datetime

from CNNModel.paso3_training import (
    CNN_DualBranch_Mejorado,
    DEAMDataset
)
from EmotionDetection.Metric.evaluation_metric import evaluate_colors_hybrid_improved

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, config=None):
    """
    Carga modelo entrenado desde archivo .pth
    
    Args:
        model_path: ruta al archivo del modelo
        config: configuración del modelo (si None, se extrae del checkpoint)
        
    Returns:
        model: modelo cargado en modo evaluación
        config: configuración del modelo
    """
    logger.info(f"Cargando modelo desde: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if config is None:
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

def evaluate_model(model, test_loader, config, output_dir):
    """
    Evalúa el modelo en el conjunto de test.
    
    Calcula métricas de regresión estándar y sistema de evaluación por colores (métrica híbrida).
    Genera visualizaciones y exporta resultados.
    
    Args:
        model: modelo a evaluar
        test_loader: DataLoader con datos de test
        config: configuración del modelo
        output_dir: directorio para guardar resultados
        
    Returns:
        metrics: diccionario con métricas de regresión
        color_results: resultados de evaluación por colores (métrica híbrida)
    """
    logger.info("Evaluando modelo...")
    
    # Recolectar predicciones
    predictions = {
        'valence_pred': [], 'arousal_pred': [],
        'valence_true': [], 'arousal_true': [],
        'valence_std': [], 'arousal_std': []
    }
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            spectrograms = batch['spectrogram'].to(device)
            
            # Predicciones
            valence_pred, arousal_pred = model(spectrograms)
            
            # Guardar resultados
            predictions['valence_pred'].extend(valence_pred.cpu().numpy())
            predictions['arousal_pred'].extend(arousal_pred.cpu().numpy())
            predictions['valence_true'].extend(batch['valence'].numpy())
            predictions['arousal_true'].extend(batch['arousal'].numpy())
            predictions['valence_std'].extend(batch['valence_std'].numpy())
            predictions['arousal_std'].extend(batch['arousal_std'].numpy())
    
    # Convertir a numpy arrays
    for key in predictions:
        predictions[key] = np.array(predictions[key])
    
    # Evaluación con sistema de colores (métrica híbrida)
    color_results = evaluate_colors_hybrid_improved(
        v_pred=predictions['valence_pred'],
        a_pred=predictions['arousal_pred'],
        v_true=predictions['valence_true'],
        a_true=predictions['arousal_true'],
        v_std_individual=predictions['valence_std'],
        a_std_individual=predictions['arousal_std'],
        threshold=config.get('threshold', 0.25)
    )
    
    # Métricas estándar de regresión
    metrics = {
        'valence': {
            'mse': mean_squared_error(predictions['valence_true'], predictions['valence_pred']),
            'mae': mean_absolute_error(predictions['valence_true'], predictions['valence_pred']),
            'r2': r2_score(predictions['valence_true'], predictions['valence_pred'])
        },
        'arousal': {
            'mse': mean_squared_error(predictions['arousal_true'], predictions['arousal_pred']),
            'mae': mean_absolute_error(predictions['arousal_true'], predictions['arousal_pred']),
            'r2': r2_score(predictions['arousal_true'], predictions['arousal_pred'])
        }
    }
    
    create_evaluation_plots(predictions, color_results, metrics, output_dir)
    save_results(metrics, color_results, output_dir)
    export_predictions_csv(predictions, color_results, output_dir)
    create_scatter_predicciones_pdf(predictions, color_results, output_dir)
    
    return metrics, color_results

def create_evaluation_plots(predictions, color_results, metrics, output_dir):
    """
    Crea visualizaciones de evaluación del modelo.
    
    Genera scatter plot comparando ground truth vs predicciones con
    clasificación por colores (métrica híbrida), distribución por colores
    y métricas de regresión estándar.
    """
    logger.info("Creando visualizaciones...")
    
    # Scatter plot con sistema de evaluación por colores (métrica híbrida)
    plt.figure(figsize=(20, 8))
    
    # Subplot 1: Ground Truth
    plt.subplot(121)
    plt.scatter(predictions['valence_true'], predictions['arousal_true'],
               c='lightblue', alpha=0.6, s=25,
               label=f'Ground Truth ({len(predictions["valence_true"])} muestras)')
    
    plt.axhline(y=5, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=5, color='black', linestyle='--', alpha=0.3)
    plt.xlim(1, 9)
    plt.ylim(1, 9)
    plt.xlabel('Valence Real')
    plt.ylabel('Arousal Real')
    plt.title('Ground Truth (TEST SET)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Predicciones con colores
    plt.subplot(122)
    plt.scatter(predictions['valence_pred'][color_results['red_mask']],
               predictions['arousal_pred'][color_results['red_mask']],
               c='red', alpha=0.7, s=25,
               label=f'Rojo ({color_results["red_cases"]})')
    
    plt.scatter(predictions['valence_pred'][color_results['orange_mask']],
               predictions['arousal_pred'][color_results['orange_mask']],
               c='orange', alpha=0.7, s=25,
               label=f'Naranja ({color_results["orange_cases"]})')
    
    plt.scatter(predictions['valence_pred'][color_results['yellow_mask']],
               predictions['arousal_pred'][color_results['yellow_mask']],
               c='gold', alpha=0.7, s=25,
               label=f'Amarillo ({color_results["yellow_cases"]})')
    
    plt.scatter(predictions['valence_pred'][color_results['green_mask']],
               predictions['arousal_pred'][color_results['green_mask']],
               c='green', alpha=0.7, s=25,
               label=f'Verde ({color_results["green_cases"]})')
    
    plt.axhline(y=5, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=5, color='black', linestyle='--', alpha=0.3)
    plt.plot([1, 9], [1, 9], 'k--', alpha=0.3, label='Diagonal perfecta')
    
    plt.xlim(1, 9)
    plt.ylim(1, 9)
    plt.xlabel('Valence Predicho')
    plt.ylabel('Arousal Predicho')
    
    # Título con métricas
    total_samples = len(predictions['valence_pred'])
    title = f'Predicciones del Modelo\n'
    title += f'Rojos: {color_results["red_cases"]} ({color_results["red_cases"]/total_samples*100:.1f}%) | '
    title += f'Mejora vs modelo anterior: -{221-color_results["red_cases"]} casos rojos\n'
    title += f'Correlación V-A: {color_results["pred_correlation"]:.3f}'
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Evaluación del Modelo - Test Set', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Distribución por colores y métricas de regresión
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Distribución por colores
    plt.subplot(121)
    colors_data = [
        color_results['red_cases'],
        color_results['orange_cases'], 
        color_results['yellow_cases'],
        color_results['green_cases']
    ]
    colors_labels = ['Rojo', 'Naranja', 'Amarillo', 'Verde']
    colors_palette = ['red', 'orange', 'gold', 'green']
    
    wedges, texts, autotexts = plt.pie(colors_data, labels=colors_labels, colors=colors_palette,
                                      autopct='%1.1f%%', startangle=90, explode=(0.05, 0, 0, 0))
    
    # Mejorar el formato del texto
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.title(f'Distribución por Colores\nTotal: {sum(colors_data)} muestras', fontsize=12, pad=20)
    
    # Subplot 2: Métricas de regresión
    plt.subplot(122)
    metrics_labels = ['MSE', 'MAE', 'R²']
    valence_metrics = [metrics['valence'][m.lower()] for m in ['mse', 'mae', 'r2']]
    arousal_metrics = [metrics['arousal'][m.lower()] for m in ['mse', 'mae', 'r2']]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    plt.bar(x - width/2, valence_metrics, width, label='Valence', alpha=0.7, color='blue')
    plt.bar(x + width/2, arousal_metrics, width, label='Arousal', alpha=0.7, color='orange')
    
    plt.xlabel('Métrica')
    plt.ylabel('Valor')
    plt.title('Métricas de Regresión')
    plt.xticks(x, metrics_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Evaluación Completa del Modelo', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_complete.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_results(metrics, color_results, output_dir):
    """
    Guarda resultados de evaluación en archivos JSON y texto.
    
    Exporta métricas de regresión estándar, análisis con sistema de colores (métrica híbrida) y comparación con baseline.
    """
    import json
    from datetime import datetime
    
    results = {
        'evaluation_date': datetime.now().isoformat(),
        'metrics': metrics,
        'color_analysis': {
            'total_samples': int(color_results['red_cases'] + color_results['orange_cases'] + 
                               color_results['yellow_cases'] + color_results['green_cases']),
            'red_cases': int(color_results['red_cases']),
            'orange_cases': int(color_results['orange_cases']),
            'yellow_cases': int(color_results['yellow_cases']),
            'green_cases': int(color_results['green_cases']),
            'pred_correlation': float(color_results['pred_correlation']),
            'baseline_improvement': {
                'previous_model_red_cases': 221,
                'current_red_cases': int(color_results['red_cases']),
                'improvement_cases': 221 - int(color_results['red_cases']),
                'improvement_percentage': ((221 - color_results['red_cases']) / 221) * 100
            }
        }
    }
    
    # Guardar JSON
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Guardar resumen en texto
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write("="*50 + "\n")
        f.write("RESUMEN DE EVALUACIÓN\n")
        f.write("="*50 + "\n\n")
        
        f.write("SISTEMA DE COLORES (MÉTRICA HÍBRIDA):\n")
        f.write(f"- Rojos: {results['color_analysis']['red_cases']} casos\n")
        f.write(f"- Naranjas: {results['color_analysis']['orange_cases']} casos\n")
        f.write(f"- Amarillos: {results['color_analysis']['yellow_cases']} casos\n")
        f.write(f"- Verdes: {results['color_analysis']['green_cases']} casos\n\n")
        
        f.write("MÉTRICAS VALENCE:\n")
        f.write(f"- MSE: {metrics['valence']['mse']:.4f}\n")
        f.write(f"- MAE: {metrics['valence']['mae']:.4f}\n")
        f.write(f"- R²: {metrics['valence']['r2']:.4f}\n\n")
        
        f.write("MÉTRICAS AROUSAL:\n")
        f.write(f"- MSE: {metrics['arousal']['mse']:.4f}\n")
        f.write(f"- MAE: {metrics['arousal']['mae']:.4f}\n")
        f.write(f"- R²: {metrics['arousal']['r2']:.4f}\n\n")
        
        f.write("COMPARACIÓN CON MODELO ANTERIOR:\n")
        f.write(f"- Modelo anterior (limitación): {results['color_analysis']['baseline_improvement']['previous_model_red_cases']} casos rojos\n")
        f.write(f"- Modelo actual: {results['color_analysis']['baseline_improvement']['current_red_cases']} casos rojos\n")
        f.write(f"- Reducción lograda: {results['color_analysis']['baseline_improvement']['improvement_cases']} casos\n")
        f.write(f"- Porcentaje de mejora: {results['color_analysis']['baseline_improvement']['improvement_percentage']:.1f}%\n\n")
        
        # Agregar precisión del modelo
        casos_correctos = (results['color_analysis']['green_cases'] + 
                          results['color_analysis']['yellow_cases'] + 
                          results['color_analysis']['orange_cases'])
        precision_modelo = (casos_correctos / results['color_analysis']['total_samples']) * 100
        
        f.write("PRECISIÓN DEL MODELO:\n")
        f.write(f"- Casos correctos: {casos_correctos}/{results['color_analysis']['total_samples']} ({precision_modelo:.1f}%)\n")
        f.write(f"- Casos rojos (errores): {results['color_analysis']['red_cases']}/{results['color_analysis']['total_samples']} ({results['color_analysis']['red_cases']/results['color_analysis']['total_samples']*100:.1f}%)\n")

def export_predictions_csv(predictions, color_results, output_dir):
    """
    Exporta predicciones detalladas a archivo CSV.
    
    Incluye valores reales, predichos, desviaciones estándar y color asignado.
    """
    # Crear DataFrame con los resultados
    data = {
        'valence_true': predictions['valence_true'].flatten(),
        'arousal_true': predictions['arousal_true'].flatten(),
        'valence_pred': predictions['valence_pred'].flatten(),
        'arousal_pred': predictions['arousal_pred'].flatten(),
        'valence_std': predictions['valence_std'].flatten(),
        'arousal_std': predictions['arousal_std'].flatten(),
        'assigned_color': [
            'red' if m else 'orange' if o else 'yellow' if y else 'green' 
            for m, o, y in zip(
                color_results['red_mask'],
                color_results['orange_mask'],
                color_results['yellow_mask']
            )
        ]
    }
    df = pd.DataFrame(data)
    
    # Guardar CSV
    csv_path = os.path.join(output_dir, 'predictions_detail.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Predicciones detalladas guardadas en: {csv_path}")

def create_scatter_predicciones_pdf(predictions, color_results, output_dir):
    """
    Crea gráfico de dispersión de predicciones en formatos de alta calidad.
    
    Genera archivos PDF, PNG y EPS del scatter plot con sistema de evaluación
    por colores (métrica híbrida).
    """
    logger.info("Generando gráfico de dispersión de predicciones...")
    
    # Crear figura de alta calidad
    plt.figure(figsize=(10, 10))
    
    # Graficar predicciones con sistema de colores (métrica híbrida)
    plt.scatter(predictions['valence_pred'][color_results['red_mask']],
               predictions['arousal_pred'][color_results['red_mask']],
               c='red', alpha=0.7, s=25,
               label=f'Rojo ({color_results["red_cases"]})')
    
    plt.scatter(predictions['valence_pred'][color_results['orange_mask']],
               predictions['arousal_pred'][color_results['orange_mask']],
               c='orange', alpha=0.7, s=25,
               label=f'Naranja ({color_results["orange_cases"]})')
    
    plt.scatter(predictions['valence_pred'][color_results['yellow_mask']],
               predictions['arousal_pred'][color_results['yellow_mask']],
               c='gold', alpha=0.7, s=25,
               label=f'Amarillo ({color_results["yellow_cases"]})')
    
    plt.scatter(predictions['valence_pred'][color_results['green_mask']],
               predictions['arousal_pred'][color_results['green_mask']],
               c='green', alpha=0.7, s=25,
               label=f'Verde ({color_results["green_cases"]})')
    
    # Líneas de referencia y diagonal
    plt.axhline(y=5, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=5, color='black', linestyle='--', alpha=0.3)
    plt.plot([1, 9], [1, 9], 'k--', alpha=0.3, label='Diagonal perfecta')
    
    plt.xlim(1, 9)
    plt.ylim(1, 9)
    plt.xlabel('Valence Predicho', fontsize=12)
    plt.ylabel('Arousal Predicho', fontsize=12)
    
    # Título con métricas clave
    total_samples = len(predictions['valence_pred'])
    title = f'Predicciones del Modelo\n'
    title += f'Rojos: {color_results["red_cases"]} ({color_results["red_cases"]/total_samples*100:.1f}%) | '
    title += f'Mejora vs modelo anterior: -{221-color_results["red_cases"]} casos rojos\n'
    title += f'Correlación V-A: {color_results["pred_correlation"]:.3f}'
    
    plt.title(title, fontsize=13, pad=20)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Guardar en múltiples formatos de alta calidad
    scatter_pdf_path = os.path.join(output_dir, 'predicciones_scatter_individual.pdf')
    scatter_png_path = os.path.join(output_dir, 'predicciones_scatter_individual.png')
    scatter_eps_path = os.path.join(output_dir, 'predicciones_scatter_individual.eps')
    
    plt.savefig(scatter_pdf_path, bbox_inches='tight', facecolor='white', 
                format='pdf', dpi=300)
    plt.savefig(scatter_png_path, bbox_inches='tight', facecolor='white', 
                format='png', dpi=300)
    plt.savefig(scatter_eps_path, bbox_inches='tight', facecolor='white', 
                format='eps', dpi=300)
    
    logger.info(f"Gráfico individual guardado:")
    logger.info(f"   • {scatter_pdf_path}")
    logger.info(f"   • {scatter_png_path}")
    logger.info(f"   • {scatter_eps_path}")
    
    plt.close()

def main():
    """
    Función principal de evaluación.
    
    Carga el modelo, prepara el dataset de test y genera todas las
    métricas y visualizaciones de evaluación.
    """
    parser = argparse.ArgumentParser(description='Evaluación del modelo CNN dual-branch')
    parser.add_argument('--model_path', type=str, 
                       default='train_results/best_model.pth',
                       help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--test_data', type=str, 
                       default='../dataset/DEAM/test_annotations.csv',
                       help='Ruta a las anotaciones del test set')
    parser.add_argument('--spectrograms_dir', type=str,
                       default='../dataset/DEAM/mel_numpy',
                       help='Directorio de espectrogramas')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Crear directorio de resultados
    output_dir = args.output_dir or f"evaluation_results_{int(datetime.now().timestamp())}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar modelo entrenado
    model, config = load_model(args.model_path)
    
    # Preparar dataset de evaluación
    test_df = pd.read_csv(args.test_data)
    test_df.columns = test_df.columns.str.strip().str.lower()
    test_dataset = DEAMDataset(test_df, args.spectrograms_dir, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Ejecutar evaluación completa
    metrics, color_results = evaluate_model(model, test_loader, config, output_dir)
    
    # Imprimir resumen
    logger.info("\n" + "="*50)
    logger.info("RESUMEN DE EVALUACIÓN:")
    logger.info("\nSISTEMA DE COLORES (MÉTRICA HÍBRIDA):")
    total = (color_results['red_cases'] + color_results['orange_cases'] + 
             color_results['yellow_cases'] + color_results['green_cases'])
    logger.info(f"Verdes:   {color_results['green_cases']:4d} ({color_results['green_cases']/total*100:5.1f}%)")
    logger.info(f"Amarillos:{color_results['yellow_cases']:4d} ({color_results['yellow_cases']/total*100:5.1f}%)")
    logger.info(f"Naranjas: {color_results['orange_cases']:4d} ({color_results['orange_cases']/total*100:5.1f}%)")
    logger.info(f"Rojos:    {color_results['red_cases']:4d} ({color_results['red_cases']/total*100:5.1f}%)")
    logger.info(f"\nTotal muestras: {total}")
    
    logger.info(f"\nMÉTRICAS DE VALENCE:")
    logger.info(f"- MSE: {metrics['valence']['mse']:.4f}")
    logger.info(f"- MAE: {metrics['valence']['mae']:.4f}")
    logger.info(f"- R²:  {metrics['valence']['r2']:.4f}")
    
    logger.info(f"\nMÉTRICAS DE AROUSAL:")
    logger.info(f"- MSE: {metrics['arousal']['mse']:.4f}")
    logger.info(f"- MAE: {metrics['arousal']['mae']:.4f}")
    logger.info(f"- R²:  {metrics['arousal']['r2']:.4f}")
    
    logger.info(f"\nCOMPARACIÓN CON MODELO ANTERIOR:")
    logger.info(f"- Modelo anterior (limitación): 221 casos rojos")
    logger.info(f"- Modelo actual: {color_results['red_cases']} casos rojos")
    logger.info(f"- Reducción lograda: {221-color_results['red_cases']} casos ({((221-color_results['red_cases'])/221)*100:.1f}%)")
    
    # Calcular precisión del modelo
    casos_correctos = color_results['green_cases'] + color_results['yellow_cases'] + color_results['orange_cases']
    precision_modelo = (casos_correctos / total) * 100
    
    logger.info(f"\nPRECISIÓN DEL MODELO:")
    logger.info(f"- Casos correctos: {casos_correctos}/{total} ({precision_modelo:.1f}%)")
    logger.info(f"- Casos rojos (errores): {color_results['red_cases']}/{total} ({color_results['red_cases']/total*100:.1f}%)")
    
    logger.info(f"\nDIAGNÓSTICOS:")
    logger.info(f"- Correlación V-A: {color_results['pred_correlation']:.3f}")
    logger.info(f"- Std Valence: {color_results['pred_valence_std']:.3f}")
    logger.info(f"- Std Arousal: {color_results['pred_arousal_std']:.3f}")
    logger.info("="*50)
    
    logger.info(f"Evaluación completada. Resultados en: {output_dir}")

if __name__ == "__main__":
    main()
