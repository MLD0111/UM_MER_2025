#!/usr/bin/env python3
"""
Entrenamiento del modelo CNN dual-branch para predicción de valence y arousal.

El script implementa:
- Arquitectura CNN con ramas independientes para valence y arousal
- Función de pérdida que combina CCC, MAE y regularizaciones anti-correlación
- Early stopping basado en casos rojos (sistema de evaluación por colores por métrica híbrida)
- Scheduler con warm restarts para optimización gradual
- Visualizaciones de evolución del entrenamiento y predicciones
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
import logging
import random
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def convert_to_serializable(obj):
    """Convierte objetos numpy y tensores a tipos serializables por JSON"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
    
# Importar métrica híbrida
from EmotionDetection.Metric.evaluation_metric import evaluate_colors_hybrid_improved

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.environ.get('SUPPRESS_DEVICE_LOGS'):
    logger.info(f"Using device: {device}")

# ==================== CONFIGURACIÓN DEL ENTRENAMIENTO ====================
CONFIG_MEJORADO = {
    # Parámetros básicos de entrenamiento
    'batch_size': 32,
    'learning_rate': 0.0006,           # Learning rate inicial
    'min_learning_rate': 3e-7,         # Learning rate mínimo para el scheduler
    'epochs': 200,
    'patience': 35,                     # Épocas de paciencia para early stopping

    # Regularización para evitar overfitting
    'dropout_rate': 0.28,               # Tasa de dropout en capas fully connected
    'weight_decay': 2.5e-4,             # Regularización L2 en el optimizador
    'gradient_clip': 0.9,               # Clipping de gradientes para estabilidad

    # Pesos de la función de pérdida
    'loss_type': 'mejorada_anti_correlacion',
    'ccc_weight': 0.4,                  # Concordance Correlation Coefficient
    'mae_weight': 0.3,                  # Mean Absolute Error ponderado por incertidumbre
    'orthogonal_weight': 0.1,           # Penalización de correlación extrema (>0.9)
    'anti_correlation_weight': 0.15,    # Penalización de correlación alta (>0.7)
    'dispersion_weight': 0.05,          # Fomenta dispersión similar al ground truth
    
    # Configuración del scheduler de learning rate (Cosine Warm Restart)
    'scheduler_type': 'cosine_warm_restart',
    'T_0': 40,                          # Longitud del primer ciclo en épocas
    'T_mult': 1,                        # Multiplicador de longitud de ciclos
    'eta_min_factor': 0.02,             # Factor para calcular LR mínimo
    
    # Exponential Moving Average para estabilizar pesos
    'use_ema': True,
    'ema_decay': 0.995,                 # Factor de decaimiento del EMA
    
    # Inicialización de pesos
    'init_type': 'xavier_uniform',
    'init_bias_valence': 5.0,           # Bias inicial centrado en el espacio V-A
    'init_bias_arousal': 5.0,
    
    # Batch Normalization
    'batch_norm_momentum': 0.99,        # Momentum para batch normalization
    
    # Parámetros de evaluación
    'threshold': 0.25,                  # Umbral para clasificación de colores
    'target_red_cases': 130,            # Meta de casos rojos para considerar éxito
    'seed': 42,                         # Semilla para reproducibilidad
}

def set_seed(seed=42):
    """Configurar semilla para reproducibilidad en todos los generadores aleatorios"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed} for reproducibility")

class ExponentialMovingAverage:
    """
    Exponential Moving Average de los pesos del modelo.
    Mantiene un promedio móvil de los parámetros para estabilizar el entrenamiento.
    """
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class CosineWarmRestartScheduler:
    """
    Scheduler de learning rate con warm restarts usando coseno.
    Reduce gradualmente el LR en ciclos que se reinician periódicamente.
    """
    def __init__(self, optimizer, T_0=15, T_mult=1, eta_min_factor=0.1):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min_factor = eta_min_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.T_cur = 0
        self.T_i = T_0
        
    def step(self):
        self.T_cur += 1
        
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            eta_min = self.base_lrs[i] * self.eta_min_factor
            lr = eta_min + (self.base_lrs[i] - eta_min) * (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
            param_group['lr'] = lr
        
        return self.optimizer.param_groups[0]['lr']

def concordance_correlation_coefficient(y_pred, y_true):
    """
    Calcula el Concordance Correlation Coefficient (CCC).
    Métrica que evalúa la concordancia entre predicciones y valores reales.
    Retorna 1-CCC para usar como pérdida (menor CCC = mayor pérdida).
    """
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    
    var_true = torch.var(y_true)
    var_pred = torch.var(y_pred)
    
    covariance = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    
    return 1.0 - ccc

def compute_mejorada_simplificada_loss(valence_pred, arousal_pred, valence_true, arousal_true, 
                                     valence_std, arousal_std, config, epoch=0):
    """
    Función de pérdida con múltiples componentes.
    
    Componentes:
    1. CCC Loss: Concordance Correlation Coefficient para valence y arousal
    2. MAE Loss: Error absoluto medio ponderado por la incertidumbre (std)
    3. Anti-Correlation Loss: Penaliza correlaciones V-A mayores a 0.7
    4. Orthogonal Loss: Penaliza correlaciones extremas mayores a 0.9
    5. Dispersion Loss: Fomenta dispersión similar al ground truth
    
    Args:
        valence_pred, arousal_pred: predicciones del modelo
        valence_true, arousal_true: valores reales
        valence_std, arousal_std: desviaciones estándar (incertidumbre)
        config: diccionario con pesos de cada componente
        epoch: número de época actual (para logging)
        
    Returns:
        total_loss: pérdida total combinada
        red_cases: número de casos clasificados como rojos
    """
    device = valence_pred.device
    batch_size = len(valence_pred)
    
    # 1. CCC Loss
    ccc_val_loss = concordance_correlation_coefficient(valence_pred, valence_true)
    ccc_aro_loss = concordance_correlation_coefficient(arousal_pred, arousal_true)
    ccc_loss = config['ccc_weight'] * (ccc_val_loss + ccc_aro_loss)
    
    # 2. MAE Loss ponderado por incertidumbre
    val_weights = 1.0 / (valence_std + 0.1)
    aro_weights = 1.0 / (arousal_std + 0.1)
    
    mae_val = torch.mean(val_weights * torch.abs(valence_pred - valence_true))
    mae_aro = torch.mean(aro_weights * torch.abs(arousal_pred - arousal_true))
    mae_loss = config['mae_weight'] * (mae_val + mae_aro)
    
    # 3. Regularización anti-correlación
    if batch_size > 1:
        correlation = torch.corrcoef(torch.stack([valence_pred, arousal_pred]))[0, 1]
        
        # Penalizar correlaciones altas (>0.7) para evitar línea diagonal
        high_corr_penalty = torch.maximum(torch.abs(correlation) - 0.7, torch.tensor(0.0))
        anti_correlation_loss = config.get('anti_correlation_weight', 0.2) * (high_corr_penalty ** 2)
        
        # Penalización extrema para colapso total (>0.9)
        extreme_corr = torch.maximum(torch.abs(correlation) - 0.9, torch.tensor(0.0))
        orthogonal_loss = config['orthogonal_weight'] * (extreme_corr ** 2)
        
        correlation_penalty = anti_correlation_loss + orthogonal_loss
    else:
        correlation_penalty = torch.tensor(0.0, device=device)
        
    # 4. Dispersión loss - fomentar variabilidad como en ground truth
    val_std_target = 1.5
    aro_std_target = 1.3
    val_std_actual = torch.std(valence_pred) if batch_size > 1 else torch.tensor(1.0)
    aro_std_actual = torch.std(arousal_pred) if batch_size > 1 else torch.tensor(1.0)
    
    dispersion_loss = config.get('dispersion_weight', 0.1) * (
        torch.abs(val_std_actual - val_std_target) + 
        torch.abs(aro_std_actual - aro_std_target)
    )
    
    # Pérdida total
    total_loss = ccc_loss + mae_loss + correlation_penalty + dispersion_loss
    
    # Evaluación de colores para early stopping
    color_results = evaluate_colors_hybrid_improved(
        v_pred=valence_pred, a_pred=arousal_pred, v_true=valence_true, a_true=arousal_true,
        v_std_individual=valence_std, a_std_individual=arousal_std, 
        threshold=config['threshold']
    )
    
    # Logging periódico
    if epoch % 10 == 0 and epoch > 0:
        if batch_size > 1:
            corr_val = correlation.item() if hasattr(correlation, 'item') else float(correlation)
            logger.info(f"Epoch {epoch:03d} - Loss: CCC={ccc_loss:.4f}, MAE={mae_loss:.4f}, "
                       f"AntiCorr={anti_correlation_loss:.4f}, Disp={dispersion_loss:.4f}, V-A Corr={corr_val:.3f}")
        else:
            logger.info(f"Epoch {epoch:03d} - Loss: CCC={ccc_loss:.4f}, MAE={mae_loss:.4f}")
    
    return total_loss, color_results['red_cases']

class DEAMDataset(Dataset):
    """Dataset DEAM optimizado"""
    
    def __init__(self, annotations_data, spectrograms_dir, mode='train'):
        self.spectrograms_dir = spectrograms_dir
        self.mode = mode
        
        if isinstance(annotations_data, str):
            self.annotations = pd.read_csv(annotations_data)
            self.annotations.columns = self.annotations.columns.str.strip().str.lower()
        else:
            self.annotations = annotations_data.copy()
        
        required_columns = ['spectrogram', 'valence', 'arousal', 'valence_std', 'arousal_std']
        missing_columns = [col for col in required_columns if col not in self.annotations.columns]
        if missing_columns:
            raise ValueError(f"Dataset debe contener: {required_columns}")
        
        # Filtrar muestras válidas
        valid_mask = (
            (self.annotations['valence'] >= 1) &
            (self.annotations['arousal'] >= 1) &
            (self.annotations['valence'] <= 9) &
            (self.annotations['arousal'] <= 9) &
            ~((self.annotations['valence'] == 5) & (self.annotations['arousal'] == 5))
        )
        self.annotations = self.annotations[valid_mask].reset_index(drop=True)
        
        logger.info(f"Dataset {mode}: {len(self.annotations)} muestras válidas")
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        
        spectrogram_path = os.path.join(self.spectrograms_dir, row['spectrogram'])
        spectrogram = np.load(spectrogram_path)
        spectrogram = torch.FloatTensor(spectrogram)
        
        return {
            'spectrogram': spectrogram,
            'valence': torch.FloatTensor([row['valence']]),
            'arousal': torch.FloatTensor([row['arousal']]),
            'valence_std': torch.FloatTensor([row['valence_std']]),
            'arousal_std': torch.FloatTensor([row['arousal_std']])
        }

class CNN_DualBranch_Mejorado(nn.Module):
    """
    Red neuronal convolucional con arquitectura dual-branch.
    
    Características:
    - Features convolucionales compartidas para extraer representaciones del espectrograma
    - Ramas independientes para predicción de valence y arousal
    - Dropout progresivo y batch normalization para regularización
    - Inicialización Xavier Uniform para estabilidad
    """
    
    def __init__(self, config):
        super(CNN_DualBranch_Mejorado, self).__init__()
        self.config = config
        
        # Capas convolucionales compartidas
        self.shared_features = nn.Sequential(
            # Primera capa: extracción de características básicas
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=config['batch_norm_momentum']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Segunda capa: características de nivel medio
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=config['batch_norm_momentum']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            
            # Tercera capa: características de alto nivel
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96, momentum=config['batch_norm_momentum']),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),  # Reduce a tamaño fijo 2x2
            nn.Flatten(),  # Output: 96 * 2 * 2 = 384 features
        )
        
        feature_size = 384
        
        # Capa fully connected compartida
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.BatchNorm1d(256, momentum=config['batch_norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout_rate'] * 0.5)
        )
        
        # Rama independiente para valence
        self.valence_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, momentum=config['batch_norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout_rate'] * 0.8),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=config['batch_norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout_rate'] * 1.0),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, momentum=config['batch_norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout_rate'] * 0.6),
            
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )
        
        # Rama independiente para arousal
        self.arousal_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, momentum=config['batch_norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout_rate'] * 0.9),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=config['batch_norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout_rate'] * 1.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, momentum=config['batch_norm_momentum']),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout_rate'] * 0.7),
            
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )
        
        self._init_weights_conservative()
        
    def _init_weights_conservative(self):
        """
        Inicializa los pesos de la red usando Xavier Uniform.
        Los bias finales se centran en 5.0 (centro del espacio V-A).
        """
        logger.info("Aplicando inicialización conservadora (Xavier Uniform)")
        
        for module in self.shared_features.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        for branch in [self.valence_branch, self.arousal_branch]:
            for module in branch.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.8)
                    if module.bias is not None:
                        if module.out_features == 1:
                            nn.init.constant_(module.bias, 5.0)
                        else:
                            nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass de la red.
        Durante entrenamiento, agrega ruido gaussiano independiente a cada rama
        para reducir la correlación entre valence y arousal.
        """
        shared = self.shared_features(x)
        shared_fc = self.shared_fc(shared)
        
        # Agregar ruido independiente en entrenamiento para anti-correlación
        if self.training:
            noise_strength = 0.02
            valence_input = shared_fc + torch.randn_like(shared_fc) * noise_strength
            arousal_input = shared_fc + torch.randn_like(shared_fc) * noise_strength
        else:
            valence_input = shared_fc
            arousal_input = shared_fc
        
        valence = self.valence_branch(valence_input).squeeze(1)
        arousal = self.arousal_branch(arousal_input).squeeze(1)
        
        return valence, arousal

def prepare_academic_dataset_split(annotations_file, val_ratio=0.2, random_state=42):
    """
    Divide el dataset en train y validación evitando data leakage.
    
    Asegura que las aumentaciones de una canción solo aparezcan en train,
    mientras que validación contiene solo espectrogramas originales de canciones distintas.
    """
    df = pd.read_csv(annotations_file)
    df.columns = df.columns.str.strip().str.lower()
    
    required_columns = ['spectrogram', 'valence', 'arousal', 'valence_std', 'arousal_std']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV debe contener: {required_columns}")
    
    def extract_base_id(filename):
        base = filename.replace('.npy', '')
        suffixes = ['_noise', '_pitch', '_pitchstretch', '_timestretch', '_tempo']
        for suffix in suffixes:
            if base.endswith(suffix):
                base = base.replace(suffix, '')
        return base
    
    df['base_song_id'] = df['spectrogram'].apply(extract_base_id)
    df['is_original'] = df['spectrogram'].apply(lambda x: not any(aug in x for aug in ['_aug', '_time', '_pitch', '_noise']))
    
    original_songs = df[df['is_original']]['base_song_id'].unique()
    train_original_ids, val_original_ids = train_test_split(
        original_songs, test_size=val_ratio, random_state=random_state
    )
    
    val_df = df[
        (df['base_song_id'].isin(val_original_ids)) & 
        (df['is_original'] == True)
    ].reset_index(drop=True)
    
    train_df = df[
        df['base_song_id'].isin(train_original_ids)
    ].reset_index(drop=True)
    
    # Verificación
    train_base_ids = set(train_df['base_song_id'].unique())
    val_base_ids = set(val_df['base_song_id'].unique())
    overlap = train_base_ids & val_base_ids
    
    is_valid = len(overlap) == 0 and val_df['is_original'].sum() == len(val_df)
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Valid={is_valid}")
    
    return train_df, val_df, is_valid

def train_model_mejorado(model, train_loader, val_loader, config):
    """
    Entrena el modelo CNN dual-branch con early stopping basado en casos rojos.
    
    Args:
        model: modelo CNN a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        config: diccionario con configuración de entrenamiento
        
    Returns:
        model: modelo entrenado
        history: lista con métricas de cada época
        results: diccionario con resultados finales
    """
    output_dir = f"output_mejorado_v{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Resultados en: {output_dir}")
    
    # Configurar optimizador
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Configurar scheduler
    scheduler = CosineWarmRestartScheduler(
        optimizer,
        T_0=config['T_0'],
        T_mult=config['T_mult'], 
        eta_min_factor=config['eta_min_factor']
    )
    
    # Configurar EMA si está habilitado
    ema = None
    if config['use_ema']:
        ema = ExponentialMovingAverage(model, decay=config['ema_decay'])
    
    # Variables de tracking
    history = []
    best_red_count = float('inf')
    patience_counter = 0
    best_model_state = None
    best_ema_state = None
    
    start_time = time.time()
    
    logger.info("Iniciando entrenamiento")
    logger.info(f"Configuración: LR={config['learning_rate']}, Batch={config['batch_size']}, "
               f"Dropout={config['dropout_rate']}, WD={config['weight_decay']}")
    logger.info("="*80)
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_red_count = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False)
        for batch in train_pbar:
            spectrograms = batch['spectrogram'].to(device)
            valence_true = batch['valence'].squeeze().to(device)
            arousal_true = batch['arousal'].squeeze().to(device)
            valence_std = batch['valence_std'].squeeze().to(device)
            arousal_std = batch['arousal_std'].squeeze().to(device)
            
            optimizer.zero_grad()
            valence_pred, arousal_pred = model(spectrograms)
            
            loss, red_count = compute_mejorada_simplificada_loss(
                valence_pred, arousal_pred, valence_true, arousal_true,
                valence_std, arousal_std, config, epoch
            )
            
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            optimizer.step()
            
            if ema is not None:
                ema.update()
            
            train_loss += loss.item()
            train_red_count += red_count
            
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Red': red_count})
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_red_count = 0
        all_v_pred, all_a_pred = [], []
        all_v_true, all_a_true = [], []
        all_v_std, all_a_std = [], []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:03d} [Val]", leave=False)
            for batch in val_pbar:
                spectrograms = batch['spectrogram'].to(device)
                valence_true = batch['valence'].squeeze().to(device)
                arousal_true = batch['arousal'].squeeze().to(device)
                valence_std = batch['valence_std'].squeeze().to(device)
                arousal_std = batch['arousal_std'].squeeze().to(device)
                
                valence_pred, arousal_pred = model(spectrograms)
                
                loss, red_count = compute_mejorada_simplificada_loss(
                    valence_pred, arousal_pred, valence_true, arousal_true,
                    valence_std, arousal_std, config, epoch
                )
                
                val_loss += loss.item()
                val_red_count += red_count
                
                # Recopilar predicciones para métricas
                all_v_pred.extend(valence_pred.cpu().numpy())
                all_a_pred.extend(arousal_pred.cpu().numpy())
                all_v_true.extend(valence_true.cpu().numpy())
                all_a_true.extend(arousal_true.cpu().numpy())
                all_v_std.extend(valence_std.cpu().numpy())
                all_a_std.extend(arousal_std.cpu().numpy())
                
                val_pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Red': red_count})
        
        # Scheduler step
        current_lr = scheduler.step()
        
        # Métricas de época
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calcular métricas de predicciones
        pred_correlation = np.corrcoef(all_v_pred, all_a_pred)[0, 1] if len(all_v_pred) > 1 else 0
        pred_valence_std = np.std(all_v_pred)
        pred_arousal_std = np.std(all_a_pred)
        pred_valence_range = np.max(all_v_pred) - np.min(all_v_pred)
        pred_arousal_range = np.max(all_a_pred) - np.min(all_a_pred)
        
        # Detección de mode collapse
        collapse_detected = (
            abs(pred_correlation) > 0.85 or
            pred_valence_std < 0.3 or
            pred_arousal_std < 0.3 or
            pred_valence_range < 1.0 or
            pred_arousal_range < 1.0
        )
        
        # Registrar métricas de la época
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_red_count': train_red_count,
            'val_red_count': val_red_count,
            'pred_correlation': pred_correlation,
            'pred_valence_std': pred_valence_std,
            'pred_arousal_std': pred_arousal_std,
            'pred_valence_range': pred_valence_range,
            'pred_arousal_range': pred_arousal_range,
            'learning_rate': current_lr,
            'scheduler_mode': 'CosineWarmRestart',
            'collapse_detected': collapse_detected
        }
        history.append(epoch_data)
        
        # Logging por época
        train_red_pct = (train_red_count / len(train_loader.dataset)) * 100
        val_red_pct = (val_red_count / len(val_loader.dataset)) * 100
        
        logger.info(f"Época {epoch+1:03d}/{config['epochs']}")
        logger.info(f"  Loss: Train={train_loss:.4f}, Val={val_loss:.4f}")
        logger.info(f"  Casos rojos: Train={train_red_count} ({train_red_pct:.1f}%), Val={val_red_count} ({val_red_pct:.1f}%)")
        logger.info(f"  Correlación V-A: {pred_correlation:.3f} {'[COLLAPSE]' if abs(pred_correlation) > 0.85 else ''}")
        logger.info(f"  LR: {current_lr:.6f}")
        
        # Advertencia de mode collapse
        if collapse_detected:
            collapse_reasons = []
            if abs(pred_correlation) > 0.85:
                collapse_reasons.append(f"correlación={pred_correlation:.3f}")
            if pred_valence_std < 0.3:
                collapse_reasons.append(f"valence_std={pred_valence_std:.3f}")
            if pred_arousal_std < 0.3:
                collapse_reasons.append(f"arousal_std={pred_arousal_std:.3f}")
            
            logger.warning(f"Mode collapse detectado: {', '.join(collapse_reasons)}")

        # Early stopping
        if val_red_count < best_red_count:
            best_red_count = val_red_count
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            if ema is not None:
                best_ema_state = ema.shadow.copy()
            
            logger.info(f"Nuevo mejor modelo: {best_red_count} casos rojos")
            
            # Guardar checkpoint del mejor modelo
            try:
                checkpoint_path = os.path.join(output_dir, f'best_model_epoch_{epoch+1:03d}_red_{best_red_count}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_state,
                    'ema_state_dict': best_ema_state if ema is not None else None,
                    'red_count': best_red_count,
                    'val_loss': val_loss,
                    'config': config
                }, checkpoint_path)
                logger.info(f"  Guardado en: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Error guardando modelo: {e}")
        else:
            patience_counter += 1
        
        # Early stopping
        should_stop = patience_counter >= config['patience']
        
        # Tolerancia adicional para mode collapse en épocas tempranas
        if should_stop and epoch < 25 and collapse_detected:
            logger.info(f"  Continuando entrenamiento a pesar de mode collapse temporal")
            patience_counter = max(0, patience_counter - 3)
            should_stop = False
            
        if should_stop:
            logger.info(f"Early stopping: paciencia agotada en época {epoch+1}")
            break
        
        # Crear visualización cada 5 épocas
        if (epoch + 1) % 5 == 0:
            try:
                color_results = create_color_visualization_mejorado(model, val_loader, output_dir, epoch + 1, config)
            except Exception as e:
                logger.warning(f"  Error creando visualización: {e}")
        
        # Guardar checkpoint periódico cada 20 épocas
        if (epoch + 1) % 20 == 0:
            try:
                periodic_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1:03d}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_state_dict': ema.shadow if ema is not None else None,
                    'val_loss': val_loss,
                    'val_red_count': val_red_count,
                    'history': history,
                    'config': config
                }, periodic_path)
                logger.info(f"  Checkpoint guardado: {periodic_path}")
                
                history_path = os.path.join(output_dir, 'training_history_partial.json')
                with open(history_path, 'w') as f:
                    json.dump(convert_to_serializable(history), f, indent=2)
                    
            except Exception as e:
                logger.warning(f"Error guardando checkpoint: {e}")
        
        logger.info("-" * 80)
    
    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if ema is not None and best_ema_state is not None:
            ema.shadow = best_ema_state
        logger.info(f"Modelo final cargado: {best_red_count} casos rojos (mejor resultado)")
    
    total_time = time.time() - start_time
    
    # Resultados finales
    results = {
        'config': config,
        'training_time': total_time,
        'best_red_count': best_red_count,
        'total_epochs': len(history),
        'final_collapse_status': history[-1]['collapse_detected'] if history else False
    }
    
    logger.info("="*80)
    logger.info("Entrenamiento completado")
    logger.info("="*80)
    logger.info(f"Mejor resultado: {best_red_count} casos rojos")
    logger.info(f"Tiempo total: {total_time:.1f}s ({len(history)} épocas)")
    logger.info(f"Resultados en: {output_dir}")
    if history:
        final_collapse = history[-1]['collapse_detected']
        logger.info(f"Estado final: {'Mode collapse detectado' if final_collapse else 'Entrenamiento estable'}")
    logger.info("="*80)
    
    # Guardar modelo final
    try:
        final_model_path = os.path.join(output_dir, 'final_model_complete.pth')
        torch.save({
            'epoch': len(history),
            'model_state_dict': model.state_dict(),
            'best_model_state_dict': best_model_state if best_model_state is not None else model.state_dict(),
            'ema_state_dict': best_ema_state if ema is not None and best_ema_state is not None else None,
            'best_red_count': best_red_count,
            'config': config,
            'results': results
        }, final_model_path)
        logger.info(f"Modelo final guardado: {final_model_path}")
        
        simple_model_path = os.path.join(output_dir, 'best_model_mejorado.pth') 
        torch.save(best_model_state if best_model_state is not None else model.state_dict(), simple_model_path)
        logger.info(f"State dict guardado: {simple_model_path}")
        
    except Exception as e:
        logger.error(f"Error guardando modelo final: {e}")
    
    try:
        results_path = os.path.join(output_dir, 'mejorado_results.json')
        with open(results_path, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        logger.info(f"Resultados guardados: {results_path}")
        
        history_path = os.path.join(output_dir, 'training_history_mejorado.json')
        with open(history_path, 'w') as f:
            json.dump(convert_to_serializable(history), f, indent=2)
        logger.info(f"Historial guardado: {history_path}")
        
    except Exception as e:
        logger.error(f"Error guardando archivos JSON: {e}")
    
    # Crear visualizaciones finales
    logger.info("Creando gráficos de evolución...")
    plot_training_evolution_mejorado(history, output_dir)
    create_final_color_visualization_mejorado(model, val_loader, output_dir, config)
    logger.info("Visualizaciones finales completadas")
    
    return model, history, results

def create_color_visualization_mejorado(model, val_loader, output_dir, epoch, config):
    """
    Crea visualización de predicciones clasificadas por colores.
    Compara ground truth vs predicciones del modelo usando la métrica híbrida.
    """
    model.eval()
    all_v_pred, all_a_pred = [], []
    all_v_true, all_a_true = [], []
    all_v_std, all_a_std = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            spectrograms = batch['spectrogram'].to(device)
            valence_true = batch['valence'].squeeze().to(device)
            arousal_true = batch['arousal'].squeeze().to(device)
            valence_std = batch['valence_std'].squeeze().to(device)
            arousal_std = batch['arousal_std'].squeeze().to(device)
            
            valence_pred, arousal_pred = model(spectrograms)
            
            all_v_pred.extend(valence_pred.cpu().numpy())
            all_a_pred.extend(arousal_pred.cpu().numpy())
            all_v_true.extend(valence_true.cpu().numpy())
            all_a_true.extend(arousal_true.cpu().numpy())
            all_v_std.extend(valence_std.cpu().numpy())
            all_a_std.extend(arousal_std.cpu().numpy())
    
    # Evaluar colores
    color_results = evaluate_colors_hybrid_improved(
        v_pred=np.array(all_v_pred), a_pred=np.array(all_a_pred),
        v_true=np.array(all_v_true), a_true=np.array(all_a_true),
        v_std_individual=np.array(all_v_std), a_std_individual=np.array(all_a_std),
        threshold=config['threshold']
    )
    
    # Crear plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Máscaras de colores
    red_mask = color_results['red_mask']
    orange_mask = color_results['orange_mask']
    yellow_mask = color_results['yellow_mask']
    green_mask = color_results['green_mask']
    
    # SUBPLOT 1: Ground Truth
    ax1.scatter(np.array(all_v_true), np.array(all_a_true), 
               c='lightblue', alpha=0.6, s=25, label=f'Ground Truth ({len(all_v_true)} muestras)')
    
    ax1.axhline(y=5, color='black', linestyle='--', alpha=0.3)
    ax1.axvline(x=5, color='black', linestyle='--', alpha=0.3)
    
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_xlabel('Valence Real')
    ax1.set_ylabel('Arousal Real')
    ax1.set_title(f'Ground Truth (VALIDATION) - Epoch {epoch:03d}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SUBPLOT 2: Predicciones del modelo
    ax2.scatter(np.array(all_v_pred)[red_mask], np.array(all_a_pred)[red_mask], 
               c='red', alpha=0.7, s=25, label=f'Rojo ({color_results["red_cases"]})')
    ax2.scatter(np.array(all_v_pred)[orange_mask], np.array(all_a_pred)[orange_mask], 
               c='orange', alpha=0.7, s=25, label=f'Naranja ({color_results["orange_cases"]})')
    ax2.scatter(np.array(all_v_pred)[yellow_mask], np.array(all_a_pred)[yellow_mask], 
               c='gold', alpha=0.7, s=25, label=f'Amarillo ({color_results["yellow_cases"]})')
    ax2.scatter(np.array(all_v_pred)[green_mask], np.array(all_a_pred)[green_mask], 
               c='green', alpha=0.7, s=25, label=f'Verde ({color_results["green_cases"]})')
    
    ax2.axhline(y=5, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=5, color='black', linestyle='--', alpha=0.3)
    ax2.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='Diagonal perfecta')
    
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel('Valence Predicha')
    ax2.set_ylabel('Arousal Predicho')
    
    # Título con métricas clave
    title_text = f'Predicciones del Modelo - Epoch {epoch:03d}\n'
    title_text += f'V-A Corr: {color_results["pred_correlation"]:.3f} | '
    title_text += f'Std: V={color_results["pred_valence_std"]:.3f}/A={color_results["pred_arousal_std"]:.3f} | '
    title_text += f'Rango: V={color_results["pred_valence_range"]:.2f}/A={color_results["pred_arousal_range"]:.2f}'
    
    if color_results["red_cases"] <= 180:
        title_text += ' [Meta alcanzada]'
    else:
        title_text += ' [En progreso]'
    
    ax2.set_title(title_text)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Evaluación por Colores - Epoch {epoch:03d} | Total: {color_results["total"]} muestras', 
                fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Guardar
    plt.savefig(os.path.join(output_dir, f'predictions_mejorado_epoch_{epoch:03d}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_final_color_visualization_mejorado(model, val_loader, output_dir, config):
    """Crea visualización final de las predicciones del modelo"""
    logger.info("Creando visualización final...")
    
    if hasattr(model, 'ema') and model.ema is not None:
        model.ema.apply_shadow()
    
    create_color_visualization_mejorado(model, val_loader, output_dir, 999, config)
    
    if hasattr(model, 'ema') and model.ema is not None:
        model.ema.restore()

def plot_training_evolution_mejorado(history, output_dir):
    """Crea gráficos de evolución del entrenamiento"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = [h['epoch'] for h in history]
    
    # Loss
    axes[0,0].plot(epochs, [h['train_loss'] for h in history], 'b-', label='Train Loss', alpha=0.7)
    axes[0,0].plot(epochs, [h['val_loss'] for h in history], 'r-', label='Val Loss', alpha=0.7)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Evolución del Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Casos rojos
    axes[0,1].plot(epochs, [h['val_red_count'] for h in history], 'r-', linewidth=2)
    axes[0,1].axhline(y=180, color='green', linestyle='--', label='Meta (180)')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Casos Rojos')
    axes[0,1].set_title('Casos Rojos en Validación')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Correlación V-A
    axes[0,2].plot(epochs, [h['pred_correlation'] for h in history], 'purple', linewidth=2)
    axes[0,2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0,2].axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Límite colapso')
    axes[0,2].axhline(y=-0.85, color='red', linestyle='--', alpha=0.5)
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Correlación V-A')
    axes[0,2].set_title('Correlación Valence-Arousal')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Desviaciones estándar
    axes[1,0].plot(epochs, [h['pred_valence_std'] for h in history], 'blue', label='Valence Std', alpha=0.7)
    axes[1,0].plot(epochs, [h['pred_arousal_std'] for h in history], 'orange', label='Arousal Std', alpha=0.7)
    axes[1,0].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Límite colapso')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Desviación Estándar')
    axes[1,0].set_title('Dispersión de Predicciones')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Rangos
    axes[1,1].plot(epochs, [h['pred_valence_range'] for h in history], 'blue', label='Valence Range', alpha=0.7)
    axes[1,1].plot(epochs, [h['pred_arousal_range'] for h in history], 'orange', label='Arousal Range', alpha=0.7)
    axes[1,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Límite colapso')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Rango de Predicciones')
    axes[1,1].set_title('Rango de Predicciones - MEJORADO')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1,2].plot(epochs, [h['learning_rate'] for h in history], 'green', linewidth=2)
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('Learning Rate')
    axes[1,2].set_title('Learning Rate Schedule - MEJORADO')
    axes[1,2].set_yscale('log')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle('Evolución del Entrenamiento - VERSIÓN MEJORADA', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_evolution_mejorado.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """
    Función principal de entrenamiento.
    Configura los datasets, crea el modelo y ejecuta el entrenamiento.
    """
    set_seed(CONFIG_MEJORADO['seed'])
    
    # ==================== CONFIGURACIÓN DE RUTAS ====================
    # Ajustar estas rutas según la ubicación del dataset DEAM
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    annotations_file = os.path.join(project_root, "dataset", "DEAM", "spectrogram_annotations.csv")
    spectrograms_dir = os.path.join(project_root, "dataset", "DEAM", "mel_numpy")
    
    logger.info("Preparando dataset...")
    train_df, val_df, is_valid_split = prepare_academic_dataset_split(
        annotations_file, 
        val_ratio=0.2, 
        random_state=CONFIG_MEJORADO['seed']
    )
    
    if not is_valid_split:
        logger.error("Error: Split de dataset inválido")
        return
    
    # Crear datasets
    train_dataset = DEAMDataset(train_df, spectrograms_dir, mode='train')
    val_dataset = DEAMDataset(val_df, spectrograms_dir, mode='val')
    
    # Crear data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG_MEJORADO['batch_size'],
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG_MEJORADO['batch_size'],
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    logger.info(f"Batch size: {CONFIG_MEJORADO['batch_size']}")
    logger.info(f"Learning rate: {CONFIG_MEJORADO['learning_rate']}")
    
    # Crear modelo
    model = CNN_DualBranch_Mejorado(CONFIG_MEJORADO).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modelo: {total_params:,} parámetros entrenables")
    
    # Entrenar modelo
    try:
        model, history, results = train_model_mejorado(
            model, train_loader, val_loader, CONFIG_MEJORADO
        )
        
        # Evaluación final con métrica híbrida
        logger.info("Evaluación final...")
        model.eval()
        final_red_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                spectrograms = batch['spectrogram'].to(device)
                valence_true = batch['valence'].squeeze().to(device)
                arousal_true = batch['arousal'].squeeze().to(device)
                valence_std = batch['valence_std'].squeeze().to(device)
                arousal_std = batch['arousal_std'].squeeze().to(device)
                
                valence_pred, arousal_pred = model(spectrograms)
                
                color_results = evaluate_colors_hybrid_improved(
                    v_pred=valence_pred, a_pred=arousal_pred, 
                    v_true=valence_true, a_true=arousal_true,
                    v_std_individual=valence_std, a_std_individual=arousal_std,
                    threshold=CONFIG_MEJORADO['threshold']
                )
                final_red_count += color_results['red_cases']
        
        target_achieved = final_red_count < CONFIG_MEJORADO['target_red_cases']
        logger.info(f"Meta de {CONFIG_MEJORADO['target_red_cases']} casos rojos: {'alcanzada' if target_achieved else 'no alcanzada'}")
        logger.info(f"Casos rojos finales: {final_red_count}")
        logger.info("Entrenamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main()
