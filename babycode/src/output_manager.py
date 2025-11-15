import pandas as pd
import os
import logging
from datetime import datetime
from .config import STUDY_NAME
from .config import BUCKET_NAME
import json

logger = logging.getLogger(__name__)

#-----------------------------> Guardar resultados del testeo del modelo intermedio

def guardar_resultados_test(resultados_test:dict, archivo_base=None):

    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    """

    if archivo_base is None:
        archivo_base = STUDY_NAME

    # Nombre del archivo único para todas las iteraciones
    resultados_dir = os.path.join(BUCKET_NAME, "resultados")
    os.makedirs(resultados_dir, exist_ok=True)

    # Nombre del archivo único para todas las iteraciones
    archivo = os.path.join(resultados_dir, f"{archivo_base}_test_results.json")
  
     
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
     # Agregar nueva iteración
    datos_existentes.append(resultados_test)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Resultados guardados en {archivo} - Ganancia: {resultados_test['ganancia_test_promedio']:,.0f}")


#-----------------------------> Guardar predicciones finales

def guardar_predicciones_finales(resultados_df: pl.DataFrame, nombre_archivo=None) -> str:
    """
    Guarda las predicciones finales en un archivo CSV en la carpeta predict.
    
    Args:
        resultados_df: DataFrame con numero_cliente y predict
        nombre_archivo: Nombre del archivo (si es None, usa STUDY_NAME)
    
    Returns:
        str: Ruta del archivo guardado
    """
    
    # Crear carpeta predict si no existe
    predict_dir = os.path.join(BUCKET_NAME, "predict")
    os.makedirs(predict_dir, exist_ok=True)
     
    # Definir nombre del archivo
    if nombre_archivo is None:
        nombre_archivo = STUDY_NAME
    
    # Agregar timestamp para evitar sobrescribir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_archivo = os.path.join(predict_dir, f"{nombre_archivo}_{timestamp}.csv")
    
    # Validar formato del DataFrame
    if not isinstance(resultados_df, pl.DataFrame):
        raise TypeError("resultados_df debe ser un polars DataFrame")
    
    columnas_requeridas = ['numero_de_cliente', 'Predict']
    columnas_faltantes = [col for col in columnas_requeridas if col not in resultados_df.columns]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas requeridas: {columnas_faltantes}")
    
    if resultados_df.height == 0:
        raise ValueError("El DataFrame de resultados está vacío")
    
    # Validar tipos de datos
    if resultados_df['numero_de_cliente'].dtype not in [pl.Int32, pl.Int64]:
        logger.warning("Convirtiendo 'numero_de_cliente' a entero")
        resultados_df = resultados_df.with_columns(pl.col('numero_de_cliente').cast(pl.Int64))
    
    if resultados_df['Predict'].dtype not in [pl.Int32, pl.Int64]:
        logger.warning("Convirtiendo 'Predict' a entero")
        resultados_df = resultados_df.with_columns(pl.col('Predict').cast(pl.Int64))
    
    # Validar valores de predict (deben ser 0 o 1)
    valores_unicos = resultados_df['Predict'].unique().to_list()
    valores_invalidos = [v for v in valores_unicos if v not in [0, 1]]
    if valores_invalidos:
        raise ValueError(f"La columna 'Predict' contiene valores inválidos: {valores_invalidos}. Solo se permiten 0 y 1")
    
    # Guardar archivo
    resultados_df.write_csv(ruta_archivo)
    
    logger.info(f"Predicciones guardadas en: {ruta_archivo}")
    logger.info(f"Formato del archivo:")
    logger.info(f"  Columnas: {resultados_df.columns}")
    logger.info(f"  Registros: {resultados_df.height:,}")
    logger.info(f"  Primeras filas:")
    logger.info(f"{resultados_df.head()}")
    
    return ruta_archivo

#-----------------------------------------------------------> guardar iteracion de la optimización bayesiana

def guardar_iteracion(trial, ganancia_maxima, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    resultados_dir = os.path.join(BUCKET_NAME, "resultados")
    os.makedirs(resultados_dir, exist_ok=True)

    # Nombre del archivo único para todas las iteraciones
    archivo = os.path.join(resultados_dir, f"{archivo_base}_iteraciones.json")

    
    # Datos de esta iteración

    params_completos = {
    **trial.params,
    'objective': 'binary',
    'metric': 'None',
    'verbosity': -1,
    'verbose': -1,
    'max_bin': 31
    }
    
    
    iteracion_data = {
        'trial_number': trial.number,
        'params': params_completos,
        'num_boost_round': trial.user_attrs.get('num_boost_round'),
        'num_boost_round_original': trial.user_attrs.get('num_boost_round_original'),
        'best_iteration': trial.user_attrs.get('best_iteration', None), 
        'value': float(ganancia_maxima),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE'} 
    
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo} - Ganancia: {ganancia_maxima:,.0f}")

