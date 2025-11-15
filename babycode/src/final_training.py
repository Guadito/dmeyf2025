# src/final_training.py
import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .config import FINAL_TRAIN, FINAL_PREDICT, SEMILLAS
from .best_params import *
from .gain_function import *
from .output_manager import *
from .loader import convertir_clase_ternaria_a_target_polars

logger = logging.getLogger(__name__)



def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.
  
    Args:
        df: DataFrame con todos los datos
  
    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDICT}")
  
    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
  
    # Datos de predicción: período FINAL_PREDIC 
    if isinstance(FINAL_TRAIN, list):
        df_train = df[df['foto_mes'].isin(FINAL_TRAIN)]
    else: df_train = df[df['foto_mes'] == FINAL_TRAIN]
    
    predict_data = df[df['foto_mes'] == FINAL_PREDICT]

    logger.info(f"Registros de entrenamiento: {len(df_train):,}")
    logger.info(f"Registros de predicción: {len(predict_data):,}")
  
    # Corroborar que no estén vacíos los df
    if df_train.empty:
        raise ValueError(f"No se encontraron datos de entrenamiento para foto_mes: {FINAL_TRAIN}")
    if predict_data.empty:
        raise ValueError(f"No se encontraron datos de predicción para foto_mes: {FINAL_PREDICT}")

    logger.info("Validación exitosa: ambos dataframes contienen datos")

    df_train = convertir_clase_ternaria_a_target_polars(df_train, baja_2_1=True) # Entreno el modelo con Baja+1 y Baja+2 == 1

    # Preparar features y target para entrenamiento
    X_train = df_train.drop(columns = ['clase_ternaria'])
    y_train = df_train['clase_ternaria']

    X_predict = predict_data.drop(columns = ['clase_ternaria'])
    clientes_predict = predict_data['numero_de_cliente']


    # Información sobre features y distribución
    logger.info(f"Features utilizadas: {X_train.shape[1]}")
    logger.info(f"Distribución del target en entrenamiento:")
    
        # Contar cada clase
    value_counts = y_train.value_counts()
    for clase, count in value_counts.items():
        logger.info(f"  Clase {clase}: {count:,} ({count/len(y_train)*100:.2f}%)")
  


    return X_train, y_train, X_predict, clientes_predict


#-----------------------------------------> entrenar modelo final

def entrenar_modelo_final_semillerio(X_train: pl.DataFrame, y_train: pd.Series, mejores_params: dict) -> list:
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
            'metric': None,  
            'random_state': semilla,
            'verbosity': -1,
            **mejores_params,           
        }
        

        # Normalización si hubo undersampling en la optimización bayesiana
        if undersampling_ratio != 1.0 and 'min_data_in_leaf' in params:
            valor_original = params['min_data_in_leaf']
            params['min_data_in_leaf'] = max(1, round(valor_original / undersampling_ratio))
            logger.info(f"Ajustando min_data_in_leaf: {valor_original} -> {params['min_data_in_leaf']} (factor {undersampling_ratio})")
        
        # Crear dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Copiar los parámetros para no modificar el dict original 
        params_copy = params.copy()
        num_boost_round = num_boost_round = params_copy.pop('best_iteration', params_copy.pop('num_boost_round', 200))
        
        # Entrenar modelo
        modelo = lgb.train(
            params_copy,
            train_data,
            feval=ganancia_ordenada,
            num_boost_round=num_boost_round
        )
        
        modelos.append(modelo)
        logger.info(f"Modelo {idx+1} entrenado exitosamente")
    
    logger.info(f"Total de modelos entrenados: {len(modelos)}")

    return modelos


#----------------------------------------> generar predicciones finales

def generar_predicciones_finales_por_umbral(
    modelos: list,
    X_predict: pd.DataFrame,
    clientes_predict: np.ndarray,
    umbrales: list[float],
    nombre_base: str = None) -> dict:
    """
    Genera las predicciones finales promediando varios modelos y aplica distintos umbrales
    para crear y guardar DataFrames de resultados.
    
    Args:
        modelos: Lista de modelos entrenados
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        umbrales: Lista de umbrales a aplicar
        nombre_base: Nombre base para los archivos CSV (opcional)
    
    Returns:
        dict con:
            - 'probabilidades_promedio': np.ndarray
            - 'resultados_por_umbral': {umbral: {'ruta': str, 'positivos': int, 'porcentaje': float}}
    """

    logger.info(f"Generando predicciones finales con el promedio de {len(modelos)} modelos...")

    try:
        # Generar probabilidades con cada modelo y promediarlas
        probabilidades_todos = []
        for idx, modelo in enumerate(modelos):
            proba = modelo.predict(X_predict)
            probabilidades_todos.append(proba)
        probabilidades_promedio = np.mean(probabilidades_todos, axis=0)

        # Aplicar umbrales y guardar resultados
        resultados_por_umbral = {}
        for umbral in umbrales:
            try: 
                pred_bin = (probabilidades_promedio >= umbral).astype(int)
                resultados_df = pd.DataFrame({
                    'numero_de_cliente': clientes_predict,
                    'Predict': pred_bin
                })

                # Registrar estadísticas
                positivos = int(resultados_df['Predict'].sum())
                total = len(resultados_df)
                porcentaje = round((positivos / total) * 100 if total > 0 else 0,2)
                logger.info(f"Umbral {umbral}, {positivos:,} positivos ({porcentaje:.2f}%)")

                # Guardar resultados en CSV
                nombre_archivo = f"{nombre_base or STUDY_NAME}_umbral_{str(umbral).replace('.', '_')}"
                ruta = guardar_predicciones_finales(resultados_df, nombre_archivo=nombre_archivo)


                resultados_por_umbral[umbral] = {
                    'ruta': ruta,
                    'positivos': int(positivos),
                    'porcentaje': round(porcentaje, 2)
                }

            except Exception as e:
                logger.error(f"Error al procesar umbral {umbral}: {e}", exc_info=True)
                continue  # sigue con el siguiente umbral 


        return {
            'probabilidades_promedio': probabilidades_promedio,
            'resultados_por_umbral': resultados_por_umbral
        }

    except Exception as e:
        logger.error(f"Error al generar predicciones finales: {e}", exc_info=True)
        raise


#---------------------------------> generar predicciones por cantidad de envíos

def generar_predicciones_por_cantidad(
    modelos: list,
    X_predict: pd.DataFrame,
    clientes_predict: np.ndarray,
    cantidades: list[int],  
    nombre_base: str = None) -> dict:
    """
    Genera predicciones finales seleccionando un NÚMERO FIJO de clientes
    con las probabilidades más altas.
    """
    logger.info(f"Generando predicciones finales por cantidad con {len(modelos)} modelos...")

    try:
        # Promediar las probabilidades de los modelos y aplicar  cortes por cantidad
        probabilidades_todos = [modelo.predict(X_predict) for modelo in modelos]
        probabilidades_promedio = np.mean(probabilidades_todos, axis=0)
        resultados_por_cantidad = {}
        for cantidad in cantidades:
            try:
                indices_ordenados = np.argsort(probabilidades_promedio)[::-1]
                indices_top_n = indices_ordenados[:cantidad]
                pred_bin = np.zeros_like(probabilidades_promedio, dtype=int)
                pred_bin[indices_top_n] = 1

                resultados_df = pd.DataFrame({
                    'numero_de_cliente': clientes_predict,
                    'Predict': pred_bin
                })


                positivos = int(resultados_df['Predict'].sum())
                porcentaje = round((positivos / len(resultados_df)) * 100, 2)
                logger.info(f"Cantidad {cantidad}, {positivos:,} positivos ({porcentaje:.2f}%)") # <--- CAMBIO 3

                # Guardar resultados en CSV
                nombre_archivo = f"{nombre_base or STUDY_NAME}_cantidad_{cantidad}" # <--- CAMBIO 4
                ruta = guardar_predicciones_finales(resultados_df, nombre_archivo=nombre_archivo)

                resultados_por_cantidad[cantidad] = {
                    'ruta': ruta,
                    'positivos': positivos,
                    'porcentaje': porcentaje
                }

            except Exception as e:
                logger.error(f"Error al procesar cantidad {cantidad}: {e}", exc_info=True)
                continue

        return {
            'probabilidades_promedio': probabilidades_promedio,
            'resultados_por_cantidad': resultados_por_cantidad # <--- CAMBIO 5
        }

    except Exception as e:
        logger.error(f"Error al generar predicciones finales: {e}", exc_info=True)
        raise


#-----------------------------------------> entrenar modelo final

def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> list:
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
    undersampling_ratio = PARAMETROS_LGBM['undersampling']
    
    for idx, semilla in enumerate(semillas):
        logger.info(f"Entrenando modelo {idx+1}/{len(semillas)} con semilla {semilla}")
        
        # Configurar parámetros con la semilla actual
        params = {
            'objective': 'binary',
            'metric': None,  
            'random_state': semilla,
            'verbosity': -1,
            **mejores_params,           
        }
        

        # Normalización si hubo undersampling en la optimización bayesiana
        if undersampling_ratio != 1.0 and 'min_data_in_leaf' in params:
            valor_original = params['min_data_in_leaf']
            params['min_data_in_leaf'] = max(1, round(valor_original / undersampling_ratio))
            logger.info(f"Ajustando min_data_in_leaf: {valor_original} -> {params['min_data_in_leaf']} (factor {undersampling_ratio})")
        
        # Crear dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Copiar los parámetros para no modificar el dict original 
        params_copy = params.copy()
        num_boost_round = num_boost_round = params_copy.pop('best_iteration', params_copy.pop('num_boost_round', 200))
        
        # Entrenar modelo
        modelo = lgb.train(
            params_copy,
            train_data,
            feval=ganancia_ordenada,
            num_boost_round=num_boost_round
        )
        
        modelos.append(modelo)
        logger.info(f"Modelo {idx+1} entrenado exitosamente")
    
    logger.info(f"Total de modelos entrenados: {len(modelos)}")

    return modelos


# ------------------------------>

def generar_predicciones_finales(modelos: list, X_predict: pd.DataFrame, clientes_predict: np.ndarray, corte: int = 10000) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo.
  
    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        umbral: Umbral para clasificación binaria
  
    Returns:
        pd.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info(f"Generando predicciones finales con {len(modelos)} modelos.")
  
    # Generar probabilidades con el modelo entrenado

    probabilidades_todos = []
    for idx, modelo in enumerate(modelos):
        proba = modelo.predict(X_predict)
        probabilidades_todos.append(proba)
        logger.debug(f"Predicciones del modelo {idx+1} generadas")


    # Promedio de probabilidades
    probabilidades_promedio = np.mean(probabilidades_todos, axis=0)
  
    # Convertir a predicciones binarias con el umbral establecido
    predicciones_binarias = (probabilidades_promedio >= umbral).astype(int)
  
    # Crear DataFrame de resultados
    resultados = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'Predict': predicciones_binarias})
    
    # Estadísticas
    total_predicciones = len(resultados)
    predicciones_positivas = (resultados['Predict'] == 1).sum()
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
    
    logger.info(f"Predicciones generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
    logger.info(f"  Umbral utilizado: {umbral}")
    logger.info(f"  Modelos promediados: {len(modelos)}")
  
    return resultados