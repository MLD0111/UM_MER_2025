#!/usr/bin/env python3
"""
Métrica híbrida de evaluación para clasificación emocional en espacio valence-arousal.
Prioriza correcta interpretación de cuadrante emocional sobre precisión numérica absoluta.

Clasificación por colores:
- ROJO: Cuadrante incorrecto (error interpretativo grave)
- NARANJA: Cuadrante correcto pero impreciso 
- AMARILLO: Cuadrante correcto + precisión parcial
- VERDE: Cuadrante correcto + alta precisión
"""

import numpy as np

def evaluate_colors_hybrid_improved(predictions=None, true_values=None, stds=None,
                                   v_pred=None, a_pred=None, v_true=None, a_true=None, 
                                   v_std_individual=None, a_std_individual=None,
                                   threshold=0.25):
    """
    Evaluación con métrica híbrida que prioriza interpretación emocional.
    
    Formatos de entrada:
    1. evaluate_colors_hybrid_improved(predictions, true_values, stds)
    2. evaluate_colors_hybrid_improved(v_pred=v_pred, a_pred=a_pred, v_true=v_true, ...)
    
    Clasificación:
    - ROJO: Cuadrante incorrecto
    - NARANJA: Cuadrante correcto, baja precisión
    - AMARILLO: Cuadrante correcto, precisión parcial
    - VERDE: Cuadrante correcto, alta precisión
    """
    
    # Procesar entrada según formato
    if predictions is not None and true_values is not None and stds is not None:
        # Formato matricial: convertir tensors CUDA a numpy si es necesario
        if hasattr(predictions, 'cpu'):
            predictions = predictions.detach().cpu().numpy()
        else:
            predictions = np.array(predictions)
            
        if hasattr(true_values, 'cpu'):
            true_values = true_values.detach().cpu().numpy()
        else:
            true_values = np.array(true_values)
            
        if hasattr(stds, 'cpu'):
            stds = stds.detach().cpu().numpy()
        else:
            stds = np.array(stds)
        
        v_pred = predictions[:, 0]
        a_pred = predictions[:, 1]
        v_true = true_values[:, 0]
        a_true = true_values[:, 1]
        v_std_individual = stds[:, 0]
        a_std_individual = stds[:, 1]
    else:
        # Formato: arrays individuales - manejar tensors CUDA
        def convert_to_numpy(tensor):
            if hasattr(tensor, 'cpu'):
                return tensor.detach().cpu().numpy().flatten()
            else:
                return np.array(tensor).flatten()
        
        v_pred = convert_to_numpy(v_pred)
        a_pred = convert_to_numpy(a_pred)
        v_true = convert_to_numpy(v_true)
        a_true = convert_to_numpy(a_true)
        v_std_individual = convert_to_numpy(v_std_individual)
        a_std_individual = convert_to_numpy(a_std_individual)
    
    # 1. Cuadrante correcto (criterio primario)
    def get_quadrant(v, a):
        return 2 * (v > 5.0) + (a > 5.0)
    
    pred_quads = get_quadrant(v_pred, a_pred)
    true_quads = get_quadrant(v_true, a_true)
    correct_quadrant = (pred_quads == true_quads)
    
    # 2. Dentro del intervalo de confianza individual
    v_within_individual_interval = np.abs(v_pred - v_true) <= v_std_individual
    a_within_individual_interval = np.abs(a_pred - a_true) <= a_std_individual
    within_individual_interval = v_within_individual_interval & a_within_individual_interval
    
    # 3. Error relativo según métrica
    # Ejemplo: arousal_real= 6.4, predicción= 6.8
    # → error relativo = |3.3-8.7|/|4| × 100 = 5.4/4 × 100 = 135%
    # → error relativo = |4.4-7.4|/|4| × 100 = 3.0/4 × 100 = 75%
    # → error relativo = |4.0-2.4|/|4| × 100 = 1.6/4 × 100 = 40%
    # → error relativo = |5.1-6.1|/|4| × 100 = 1.0/4 × 100 = 25%
    # → error relativo = |4.8-5.3|/|4| × 100 = 0.5/4 × 100 = 12.5%
    # → error relativo = |1.3-1.2|/|4| × 100 = 0.1/4 × 100 = 2.5%

    # Umbral 25% = 0.25 (acepta hasta 25% de error relativo) - usado en entrenamiento
    v_error_relative = np.abs(v_pred - v_true) / 4.0
    a_error_relative = np.abs(a_pred - a_true) / 4.0
    
    # Precisión: error relativo <= umbral (25%)
    v_within_25 = v_error_relative <= threshold
    a_within_25 = a_error_relative <= threshold
    
    # 4. CLASIFICACIÓN POR COLORES
    
    # ROJO: Solo si cuadrante incorrecto (error interpretativo grave)
    rojo = ~correct_quadrant
    
    # Para casos con cuadrante correcto, evaluar precisión:
    # VERDE: Cuadrante correcto + dentro intervalo + alta precisión (ambos ≤25%)
    verde = correct_quadrant & within_individual_interval & v_within_25 & a_within_25
    
    # AMARILLO: Cuadrante correcto + dentro intervalo + precisión parcial (solo uno ≤25%)  
    amarillo = correct_quadrant & within_individual_interval & (v_within_25 ^ a_within_25)  # XOR
    
    # NARANJA: Cuadrante correcto pero impreciso
    # - Fuera del intervalo OR dentro del intervalo pero baja precisión
    naranja = correct_quadrant & ~verde & ~amarillo
    
    return {
        'red_cases': np.sum(rojo),
        'orange_cases': np.sum(naranja),
        'yellow_cases': np.sum(amarillo),
        'green_cases': np.sum(verde),
        'total': len(v_pred),
        'correlation': np.corrcoef(v_pred, v_true)[0, 1] if len(v_pred) > 1 else 0.0,  # Solo para compatibilidad
        'pred_correlation': np.corrcoef(v_pred, a_pred)[0, 1] if len(v_pred) > 1 else 0.0,  # MODE COLLAPSE: correlación entre predicciones
        'pred_valence_std': np.std(v_pred),  # MODE COLLAPSE: dispersión valence
        'pred_arousal_std': np.std(a_pred),  # MODE COLLAPSE: dispersión arousal
        'pred_valence_range': np.max(v_pred) - np.min(v_pred),  # MODE COLLAPSE: rango valence
        'pred_arousal_range': np.max(a_pred) - np.min(a_pred),  # MODE COLLAPSE: rango arousal
        # Agregar máscaras individuales para visualización
        'red_mask': rojo,
        'orange_mask': naranja,
        'yellow_mask': amarillo,
        'green_mask': verde
    }


