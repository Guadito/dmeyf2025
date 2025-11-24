import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
from itertools import combinations
from scipy.stats import wilcoxon
import json
import os
from datetime import datetime
from .config import *
from .gain_function import *
from .output_manager import *
from .grafico_test import *
from .loader import *
from .features import aplicar_undersampling_clase0
import gc
import polars as pl
logger = logging.getLogger(__name__)


#---------------------------->

def create_canaritos(df: pl.DataFrame, qcanaritos: int = 5) -> pl.DataFrame:
    """
    Añade un número específico de columnas "canarito" (features aleatorias)
    a un DataFrame de Polars.

    Estas nuevas columnas contendrán valores aleatorios uniformes (entre 0 y 1)
    y se colocarán al principio del DataFrame, manteniendo el orden
    original de las demás columnas.

    Args:
        df (pl.DataFrame): El DataFrame de Polars al que se le añadirán
                           las columnas.
        qcanaritos (int): El número de columnas "canarito" que se
                            desea crear. Default = 50.

    Returns:
        pl.DataFrame: Un nuevo DataFrame con las columnas "canarito" añadidas
                      al principio.
    """

    # 1. Guardar los nombres de las columnas originales
    original_cols = df.columns
    num_filas = df.height
    
    # 2. Generar la lista de nombres para las nuevas columnas "canarito"
    canary_cols = [f"canarito_{i}" for i in range(1, qcanaritos + 1)]

    # 3. Crear las expresiones Polars para generar los números aleatorios
    canary_expressions = [pl.lit(np.random.rand(num_filas)).alias(name) for name in canary_cols]

    # 4. Añadir las nuevas columnas y reordenar todo en un solo paso
    df = df.with_columns(
        canary_expressions
    ).select(
        canary_cols + original_cols  # Concatena listas para el nuevo orden
    )

    return df

#---------------------------> carga de datos y undersampling si aplica

def preparar_datos_training_lgb(
        df: pl.DataFrame,
        training: list | int,
        validation: list | int,
        undersampling_0: int = 1,
        qcanaritos: int = 0
    ):
    """
    Prepara datos de entrenamiento y validación para LightGBM
    de forma consistente con la función de entrenamiento final.

    Args:
        df: Polars DataFrame completo
        training: lista o entero con períodos de training
        validation: lista o entero con períodos de validación
        undersampling_0: cada cuántos registros clase 0 dejar (1 = no undersampling)

    Returns:
        tuple: (lgb_train, lgb_val, X_train, y_train, X_val, y_val)
    """

    logger.info("Preparando datos para entrenamiento + validación")

    if isinstance(training, list): 
        df_train = df.filter(pl.col('foto_mes').is_in(training))
    else:
        df_train = df.filter(pl.col('foto_mes') == training)

    if isinstance(validation, list):
        df_val = df.filter(pl.col('foto_mes').is_in(validation))
    else:
        df_val = df.filter(pl.col('foto_mes') == validation)

    
    logger.info(f"Train shape: {df_train.shape[0]:,} filas × {df_train.shape[1]:,} columnas | "f"Períodos train: {training}")
    logger.info(f"Test shape: {df_val.shape[0]:,} filas × {df_val.shape[1]:,} columnas | "f"Períodos test: {validation}")


    if df_train.is_empty():
        raise ValueError(f"No se encontraron datos de training para períodos: {training}")
    if df_val.is_empty():
        raise ValueError(f"No se encontraron datos de validation para períodos: {validation}")

    
    df_train = convertir_clase_ternaria_a_target_polars(df_train, baja_2_1=True)
    df_val = convertir_clase_ternaria_a_target_polars(df_val, baja_2_1=False)

    df_train = aplicar_undersampling_clase0(df_train, undersampling_0, seed=SEMILLAS[0])
    logger.info(f"Train luego de undersampling: {len(df_train):,}")

    if qcanaritos > 0:
        df_train = create_canaritos(df_train, qcanaritos=qcanaritos)  #ver semilla
        df_val = create_canaritos(df_val, qcanaritos=qcanaritos)
    
    logger.info(f"Canaritos creados en train y test: {qcanaritos}")
    
    X_train = df_train.drop('clase_ternaria').to_pandas()
    y_train = df_train['clase_ternaria'].to_numpy()

    X_val = df_val.drop('clase_ternaria').to_pandas()
    y_val = df_val['clase_ternaria'].to_numpy()

    logger.info("Distribución training:")
    for clase, count in df_train['clase_ternaria'].value_counts().iter_rows():
        logger.info(f"  Clase {clase}: {count:,} ({count/len(df_train)*100:.0f}%)")
    


    logger.info("Distribución validation:")
    for clase, count in df_val['clase_ternaria'].value_counts().iter_rows():
        logger.info(f"  Clase {clase}: {count:,} ({count/len(df_val)*100:.0f}%)")


    lgb_train = lgb.Dataset(X_train, label=y_train)
    #lgb_val = lgb.Dataset(X_val.to_pandas(), label=y_val, reference=lgb_train)

    
    return lgb_train, X_train, y_train, X_val, y_val



# -------------------> zlightgbm

def entrenar_modelo(lgb_train: lgb.Dataset, params: dict, semillas: list, tipo: str = None) -> list:
    """
    Entrena un modelo con diferentes semillas.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        params: Hiperparámetros
        Tipo: cadena para identificar el modelo
    
    Returns:
        list: Lista de modelos entrenados
    """
    modelos = []

    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "modelos_modelos")
    os.makedirs(path_db, exist_ok=True)
    study_name = STUDY_NAME

    
    for semilla in semillas:
        logger.info(f"Entrenando modelo con semilla {semilla}")

        params["seed"] = semilla
        
        # Entrenar modelo
        modelo = lgb.train(params,lgb_train)
        modelos.append(modelo)
    
        arch_modelo = os.path.join(path_db, f"mod_{study_name}_{tipo}_{semilla}.txt")
        modelo.save_model(arch_modelo)
    
    logger.info(f"Total de modelos entrenados: {len(modelos)}")

    return modelos


#----------------------------> evaluación del modelo

def evaluar_en_test(modelos: list, X_test: pd.DataFrame, y_test: pd.Series,
                    cortes: list, corte_fijo: int = 11000) -> tuple:
    """
    Evalúa un ensemble de modelos en test.
    
    Args:
        modelos: lista de modelos LightGBM entrenados
        X_test: features de test
        y_test: target de test
        cortes: lista de cortes para calcular ganancias
        corte_fijo: corte fijo para predicción binaria
    
    Returns:
        tuple: (resultados_test, resultados_df_fijo, resultados_df, y_pred_binary_mejor, y_test, y_pred_promedio)
    """
    logger.info("INICIANDO EVALUACIÓN EN TEST")

    # Promediar predicciones de todos los modelos
    y_preds = np.array([m.predict(X_test) for m in modelos])
    y_pred_promedio = np.mean(y_preds, axis=0)

    # Ganancias por corte
    ganancias_por_corte = calcular_ganancias_por_corte(y_pred_promedio, y_test, cortes)
    mejor_corte_idx = np.argmax(ganancias_por_corte)
    mejor_corte = cortes[mejor_corte_idx]
    mejor_ganancia = ganancias_por_corte[mejor_corte_idx]

    logger.info("\n" + "=" * 80)
    logger.info("RESULTADOS POR CORTE:")
    logger.info("=" * 80)
    for corte, ganancia in zip(cortes, ganancias_por_corte):
        marcador = " MEJOR" if corte == mejor_corte else ""
        # Convertir explícitamente a float/int de Python
        corte_val = int(corte)
        ganancia_val = float(ganancia)
        logger.info(f"  Corte {corte_val:6d}: Ganancia = {ganancia_val:12,.0f}{marcador}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"MEJOR CORTE: {int(mejor_corte):,} envíos")
    logger.info(f"MEJOR GANANCIA: ${float(mejor_ganancia):,.0f}")
    logger.info("=" * 80)
    

    # Corte fijo
    logger.info(f"\nEvaluando con corte fijo: {corte_fijo:,}")
    indices_top_fijo = np.argsort(y_pred_promedio)[-corte_fijo:]
    y_pred_binary_fijo = np.zeros_like(y_pred_promedio, dtype=int)
    y_pred_binary_fijo[indices_top_fijo] = 1
    resultados_df_fijo = pd.DataFrame({
        'numero_de_cliente': X_test.index,  # o la columna con IDs
        'Predict': y_pred_binary_fijo
    })

    # Mejor corte
    logger.info(f"Generando predicciones con mejor corte: {mejor_corte:,}")
    indices_top_mejor = np.argsort(y_pred_promedio)[-mejor_corte:]
    y_pred_binary_mejor = np.zeros_like(y_pred_promedio, dtype=int)
    y_pred_binary_mejor[indices_top_mejor] = 1
    resultados_df = pd.DataFrame({
        'numero_de_cliente': X_test.index,
        'Predict': y_pred_binary_mejor
    })

    # Métricas
    tp = np.sum((y_pred_binary_mejor == 1) & (y_test == 1))
    fp = np.sum((y_pred_binary_mejor == 1) & (y_test == 0))
    tn = np.sum((y_pred_binary_mejor == 0) & (y_test == 0))
    fn = np.sum((y_pred_binary_mejor == 0) & (y_test == 1))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    accuracy = (tp + tn) / len(y_test)

    resultados_test = {
        'ganancia_test_promedio': float(mejor_ganancia),
        'mejor_corte_promedio': int(mejor_corte),
        'ganancias_por_corte': {int(c): float(g) for c, g in zip(cortes, ganancias_por_corte)},
        #'ganancias_por_corte': dict(zip(cortes, ganancias_por_corte)),
        'total_predicciones': len(y_test),
        'predicciones_positivas': int(np.sum(y_pred_binary_mejor == 1)),
        'porcentaje_positivas': float(np.mean(y_pred_binary_mejor) * 100),
        'verdaderos_positivos': int(tp),
        'falsos_positivos': int(fp),
        'verdaderos_negativos': int(tn),
        'falsos_negativos': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'timestamp': datetime.datetime.now().isoformat()
    }

    guardar_resultados_test(resultados_test)

    return resultados_test, resultados_df_fijo, resultados_df, y_pred_binary_mejor, y_test, y_pred_promedio, mejor_corte













































