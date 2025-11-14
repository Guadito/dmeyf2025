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

logger = logging.getLogger(__name__)


#---------------------------->

def create_canaritos(df: pl.DataFrame, qcanaritos: int = 50) -> pl.DataFrame:
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
        undersampling_0: int = 1
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

    
    logger.info(f"Tamaño original train: {len(df_train):,} | "f"Períodos train: {training}")
    logger.info(f"Tamaño val: {len(df_val):,} | "f"Períodos val: {validation}")


    if df_train.is_empty():
        raise ValueError(f"No se encontraron datos de training para períodos: {training}")
    if df_val.is_empty():
        raise ValueError(f"No se encontraron datos de validation para períodos: {validation}")

    df_train = convertir_clase_ternaria_a_target_polars(df_train, baja_2_1=True)
    df_val = convertir_clase_ternaria_a_target_polars(df_val, baja_2_1=False)

    df_train = aplicar_undersampling_clase0(df_train, undersampling_0, seed=SEMILLAS[0])

    logger.info(f"Train luego de undersampling: {len(df_train):,}")

    X_train = df_train.drop('clase_ternaria')
    y_train = df_train['clase_ternaria'].to_numpy()

    X_val = df_val.drop('clase_ternaria')
    y_val = df_val['clase_ternaria'].to_numpy()

   logger.info("Distribución training:")
    for clase, count in df_train['clase_ternaria'].value_counts().iter_rows():
        logger.info(f"  Clase {clase}: {count:,} ({count/len(df_train)*100:.0f}%)")

    logger.info("Distribución validation:")
    for clase, count in df_val['clase_ternaria'].value_counts().iter_rows():
        logger.info(f"  Clase {clase}: {count:,} ({count/len(df_val)*100:.0f}%)")

    lgb_train = lgb.Dataset(X_train.to_pandas(), label=y_train)
    lgb_val = lgb.Dataset(X_val.to_pandas(), label=y_val, reference=lgb_train)

    
    return lgb_train, lgb_val, X_train, y_train, X_val, y_val



# -------------------> zlightgbm

def entrenar_modelo(lgb_train: lgb.Dataset, lgb_val: lgb.Dataset, mejores_params: dict) -> list:
    """
    Entrena un modelo con diferentes semillas.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna
    
    Returns:
        list: Lista de modelos entrenados
    """
    logger.info("Iniciando entrenamiento de modelos finales con múltiples semillas")
    
    modelos = []
    semillas = SEMILLAS if isinstance(SEMILLAS, list) else [SEMILLAS]
    
    for idx, semilla in enumerate(semillas):
        logger.info(f"Entrenando modelo {idx+1}/{len(semillas)} con semilla {semilla}")
        
        # Configurar parámetros con la semilla actual
        params = {
            'objective': 'binary',
            'metric': Custom,  
            'random_state': semilla,
            'verbosity': -1,
            **mejores_params,           
        }
        

        # Entrenar modelo
        modelo = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val]  #VER
        )
        
        modelos.append(modelo)
        logger.info(f"Modelo {idx+1} entrenado exitosamente")
    
    logger.info(f"Total de modelos entrenados: {len(modelos)}")

    return modelos


# --------------- > evaluar modelo

def evaluar_en_test(df: pl.DataFrame,
                               mejores_params: dict,
                               cortes : list,
                               undersampling: int = 1,
                               repeticiones: int = 1,
                               ksemillerio: int = 1) -> tuple:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test
    usando múltiples semillas y repeticiones (ensemble) y calcula ganancias.
    
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
        undersampling: factor de undersampling de clase 0
        repeticiones: número de repeticiones del ensemble
        ksemillerio: número de semillas por repetición
    
    Returns:
        tuple: (resultados_test, y_pred_binary, y_test, y_pred_prob_promedio)
    """
    logger.info(f"INICIANDO EVALUACIÓN EN TEST")

    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "modelos_modelos")
    os.makedirs(path_db, exist_ok=True)
    study_name = STUDY_NAME


    arch_modelo = os.path.join(path_db, f"mod_{study_name}_{semilla}.txt")
            
    model = lgb.train(mejores_params, train_data)
    model.save_model(arch_modelo)

    y_pred = model.predict(X_test)

    # Calcular ganancias por corte
   try: 
       ganancias = calcular_ganancias_por_corte(y_pred, y_test, cortes)
   except Exception as e:
            logger.warning(f"No se pudo calcular la ganancia para esta repetición: {e}")
            ganancias = [np.nan] * len(cortes)  # o [0]*len(cortes) si preferís
        
    logger.info(f"Repetición {repe+1}: Ganancias por corte = {dict(zip(cortes, ganancias_ronda))}")

    # Determinar mejor corte promedio
    ganancias_promedio_por_corte = np.mean(mganancias, axis=0)
    mejor_corte_index = np.argmax(ganancias_promedio_por_corte)
    mejor_corte_cantidad = cortes[mejor_corte_index]
    mejor_ganancia_promedio = ganancias_promedio_por_corte[mejor_corte_index]

        
    # ===== Corte 1: fijo en 11.000 =====
    corte_fijo = 11000
    indices_top_fijo = np.argsort(y_pred_promedio_total)[-corte_fijo:]
    y_pred_binary_fijo = np.zeros_like(y_pred_promedio_total, dtype=int)
    y_pred_binary_fijo[indices_top_fijo] = 1

    resultados_df_fijo = pd.DataFrame({
        'numero_de_cliente': clientes_test,
        'Predict': y_pred_binary_fijo
    })

    
    # ===== Corte 2: mejor punto encontrado =====
    indices_top_mejor = np.argsort(y_pred_promedio_total)[-mejor_corte_cantidad:]
    y_pred_binary_mejor = np.zeros_like(y_pred_promedio_total, dtype=int)
    y_pred_binary_mejor[indices_top_mejor] = 1

    resultados_df = pd.DataFrame({
        'numero_de_cliente': clientes_test,
        'Predict': y_pred_binary_mejor
    })
    
    y_test_np = y_test.to_numpy() 
    
    # Calcular métricas
    total_predicciones = len(y_pred_binary_mejor)
    predicciones_positivas = np.sum(y_pred_binary_mejor == 1)
    porcentaje_positivas = predicciones_positivas / total_predicciones * 100
    verdaderos_positivos = np.sum((y_pred_binary_mejor == 1) & (y_test_np == 1))
    falsos_positivos = np.sum((y_pred_binary_mejor == 1) & (y_test_np == 0))
    verdaderos_negativos = np.sum((y_pred_binary_mejor == 0) & (y_test_np == 0))
    falsos_negativos = np.sum((y_pred_binary_mejor == 0) & (y_test_np == 1))
    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos + 1e-10)
    recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos + 1e-10)
    accuracy = (verdaderos_positivos + verdaderos_negativos) / total_predicciones

    logger.info(f"--- RESULTADO FINAL ---")
    logger.info(f"Ganancias promedio por corte: {dict(zip(cortes, ganancias_promedio_por_corte))}")
    logger.info(f"Mejor corte (promedio): {mejor_corte_cantidad} envíos")
    logger.info(f"Ganancia en mejor corte (promedio): {mejor_ganancia_promedio:,.0f}")

    resultados_test = {
        'ganancia_test_promedio': float(mejor_ganancia_promedio),
        'mejor_corte_promedio': int(mejor_corte_cantidad),
        'matriz_ganancias': mganancias.tolist(),
        'ganancias_promedio_por_corte': dict(zip(cortes, ganancias_promedio_por_corte)),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'verdaderos_positivos': int(verdaderos_positivos),
        'falsos_positivos': int(falsos_positivos),
        'verdaderos_negativos': int(verdaderos_negativos),
        'falsos_negativos': int(falsos_negativos),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'timestamp': datetime.datetime.now().isoformat()
    }

    guardar_resultados_test(resultados_test)
    

    return resultados_test, resultados_df_fijo, resultados_df, y_pred_binary_mejor, y_test, y_pred_promedio_total












































