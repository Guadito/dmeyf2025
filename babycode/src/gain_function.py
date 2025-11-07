# src/gain_function.py
import numpy as np
import pandas as pd
from .config import *
import logging
import polars as pl
import gc

logger = logging.getLogger(__name__)



def calcular_ganancia(y_true, y_pred):
    """
    Calcula la ganancia total usando la función de ganancia de la competencia.
 
    Args:
        y_true: Valores reales (0 o 1)
        y_pred: Predicciones (0 o 1)
  
    Returns:
        float: Ganancia total
    """
    # Convertir a numpy arrays si es necesario
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
  
    # Calcular ganancia vectorizada usando configuración
    # Verdaderos positivos: y_true=1 y y_pred=1 -> ganancia
    # Falsos positivos: y_true=0 y y_pred=1 -> costo
    # Verdaderos negativos y falsos negativos: ganancia = 0
  
    ganancia_total = np.sum(
        ((y_true == 1) & (y_pred == 1)) * GANANCIA_ACIERTO +  # TP
        ((y_true == 0) & (y_pred == 1)) * (-COSTO_ESTIMULO)   # FP
    )
  
    logger.debug(f"Ganancia calculada: {ganancia_total:,.0f} "
                f"(GANANCIA_ACIERTO={GANANCIA_ACIERTO}, COSTO_ESTIMULO={COSTO_ESTIMULO})")
  
    return ganancia_total


#-------------------------> Ganancia binaria para LGB

def ganancia_lgb_binary(y_pred, y_true):
    """
    Función de ganancia para LightGBM en clasificación binaria.
    Compatible con callbacks de LightGBM.
  
    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos
  
    Returns:
        tuple: (eval_name, eval_result, is_higher_better)
    """
    # Obtener labels verdaderos
    y_true_labels = y_true.get_label()
  
    # Convertir probabilidades a predicciones binarias, calcular ganancia total y devolver en formato apto LGBM
    y_pred_binary = (y_pred > 0.025).astype(int)
    ganancia_total = calcular_ganancia(y_true_labels, y_pred_binary)
    return 'ganancia', ganancia_total, True  # True = higher is better


#------------------------------------------------------> ganancia x corte

def ganancia_ordenada (y_pred, y_true) -> float:

    """
    Función de evaluación personalizada para LightGBM.
    Ordena probabilidades de mayor a menor y calcula ganancia acumulada
    para encontrar el punto de máxima ganancia.
  
    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos
  
    Returns:
        float: Ganancia total
    """

    y_true = y_true.get_label()

    # Convertir a DataFrame de Polars para procesamiento eficiente
    df_eval = pl.DataFrame({
        'y_true': y_true,
        'y_pred_proba': y_pred})

    # Ordenar por probabilidad descendente y encontrar ganancia máxima
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
    df_ordenado = df_ordenado.with_columns([pl.when(pl.col('y_true') == 1) 
                                            .then(pl.lit(780000, dtype=pl.Int64))
                                            .otherwise(pl.lit(-20000, dtype=pl.Int64))
                                            .alias('ganancia_individual')])
    df_ordenado = df_ordenado.with_columns(pl.col('ganancia_individual')
                                           .cum_sum()
                                           .alias('ganancia_acumulada'))
    ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()

    return 'ganancia', ganancia_maxima, True


#--------------------------- > ganancia acumulada meseta

def ganancia_ordenada_meseta(y_pred, y_true):
    """
    Función de evaluación personalizada para LightGBM (feval).
    Ordena probabilidades de mayor a menor y calcula ganancia acumulada
    con suavizado de meseta (rolling mean) para encontrar el punto de máxima ganancia.
  
    Args:
        y_pred: Predicciones de probabilidad del modelo
        y_true: Dataset de LightGBM con labels verdaderos
  
    Returns:
        tuple: (nombre_metrica, valor, is_higher_better)
    """
    # Obtener las etiquetas verdaderas
    y_true = y_true.get_label()
    
    # Ordenar índices por probabilidad descendente
    indices_ordenados = np.argsort(y_pred)[::-1]
    
    # Calcular ganancia individual y acumulada directamente
    ganancia_individual = np.where(y_true[indices_ordenados] == 1, 780000, -20000)
    ganancia_acumulada = np.cumsum(ganancia_individual)
    
    # Rolling mean con ventana de 2001
    window = 2001
    ganancia_meseta = np.convolve(
        ganancia_acumulada, 
        np.ones(window) / window, 
        mode='same'
    )
    
    return 'ganancia', float(ganancia_meseta.max()), True

    
# --------------------------> Ganancia para definición de umbral


def calcular_ganancia_acumulada_optimizada(y_true, y_pred_proba) -> tuple:  
    """
    Calcula la ganancia acumulada ordenando las predicciones de mayor a menor probabilidad.
    Versión optimizada para grandes datasets.
  
    Args:
        y_true: Valores verdaderos (0 o 1)
        y_pred_proba: Probabilidades predichas
  
    Returns:
        tuple: (ganancias_acumuladas, indices_ordenados, umbral_optimo)
    """
    logger.info("Calculando ganancia acumulada optimizada...")

    # Asegurar arrays posicionales
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    # Ordenar por probabilidad descendente
    indices_ordenados = np.argsort(y_pred_proba)[::-1]
    y_true_ordenado = y_true[indices_ordenados]
    y_pred_proba_ordenado = y_pred_proba[indices_ordenados]

    # Calcular ganancia acumulada vectorizada
    ganancias_individuales = np.where(y_true_ordenado == 1, GANANCIA_ACIERTO, -COSTO_ESTIMULO)
    ganancias_acumuladas = np.cumsum(ganancias_individuales)

    # Encontrar el punto de ganancia máxima
    indice_maximo = np.argmax(ganancias_acumuladas)
    umbral_optimo = y_pred_proba_ordenado[indice_maximo]

    # Cantidad de predicciones
    cantidad_predicciones = len(y_true_ordenado)

    logger.info(f"Ganancia máxima: {ganancias_acumuladas[indice_maximo]:,.0f} en posición {indice_maximo}")
    logger.info(f"Umbral óptimo: {umbral_optimo:.6f}")
    logger.info(f"Cantidad de predicciones: {cantidad_predicciones}")

    return ganancias_acumuladas, indices_ordenados, umbral_optimo