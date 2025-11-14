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

# ----------------------------> Semillerío



























# --------------- > evaluar modelo con semillerio

def evaluar_en_test_semillerio(df: pl.DataFrame,
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
    
    # Preparar datos
    if isinstance(MES_TRAIN2, list):
        df_train = df.filter(pl.col("foto_mes").is_in(MES_TRAIN2))
    else:
        df_train = df.filter(pl.col("foto_mes") == MES_TRAIN2)
    df_test = df.filter(pl.col("foto_mes") == MES_TEST)

    df_train = convertir_clase_ternaria_a_target_polars(df_train, baja_2_1=True)
    df_test = convertir_clase_ternaria_a_target_polars(df_test, baja_2_1=False)

    df_train = aplicar_undersampling_clase0(df_train, undersampling, seed=SEMILLAS[0])


    X_train = df_train.drop("clase_ternaria")
    y_train = df_train["clase_ternaria"]
    X_test = df_test.drop("clase_ternaria")
    y_test = df_test["clase_ternaria"]
    clientes_test = df_test["numero_de_cliente"].to_numpy()

    # Convertir a pandas/numpy para LightGBM
    X_train_pd = X_train.to_pandas()
    y_train_np = y_train.to_numpy()
    X_test_pd = X_test.to_pandas()   # ← mantené pandas acá
    y_test_np = y_test.to_numpy()


    train_data = lgb.Dataset(X_train_pd, label=y_train_np)
    
    # Copiar parámetros y ajustar min_child_samples
    mejores_params = mejores_params.copy()
    num_boost_round = mejores_params.pop('num_boost_round', mejores_params.pop('num_boost_round', 200))
    if 'min_child_samples' in mejores_params and 'n_train_used' in mejores_params:
        factor = len(df_train) / mejores_params['n_train_used']
        mejores_params['min_child_samples'] = int(round(mejores_params['min_child_samples'] * factor))

    # Generar semillas para ensemble
    rng = np.random.default_rng(SEMILLAS[0])
    semillas_totales = rng.choice(np.arange(100_000, 1_000_000), size=ksemillerio * repeticiones, replace=False)

    # Matriz para almacenar ganancias por repetición y corte
    mganancias = np.zeros((repeticiones, len(cortes)))
    y_pred_promedio_total = np.zeros(len(X_test))  # Para acumular probabilidades promedio de todas las repeticiones

    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "modelos_modelos")
    os.makedirs(path_db, exist_ok=True)
    study_name = STUDY_NAME

    # Loop sobre repeticiones
    for repe in range(repeticiones):
        desde = repe * ksemillerio
        hasta = (repe + 1) * ksemillerio
        semillas_ronda = semillas_totales[desde:hasta]

        y_pred_acum_ronda = np.zeros(len(X_test))
        
        # Loop sobre semillas
        for semilla in semillas_ronda:
            mejores_params['random_state'] = semilla

            arch_modelo = os.path.join(path_db, f"mod_{study_name}_{semilla}.txt")
            
            model = lgb.train(mejores_params, train_data, num_boost_round=num_boost_round)
            model.save_model(arch_modelo)

            y_pred_acum_ronda += model.predict(X_test_pd, num_iteration=num_boost_round)
            del model
            gc.collect()
        
        # Probabilidad promedio de la repetición
        y_pred_promedio_ronda = y_pred_acum_ronda / ksemillerio
        y_pred_promedio_total += y_pred_promedio_ronda / repeticiones  # Promedio final sobre todas las repeticiones

        # Calcular ganancias por corte
        try: 
            ganancias_ronda = calcular_ganancias_por_corte(y_pred_promedio_ronda, y_test, cortes)
        except Exception as e:
            logger.warning(f"No se pudo calcular la ganancia para esta repetición: {e}")
            ganancias_ronda = [np.nan] * len(cortes)  # o [0]*len(cortes) si preferís
        
        mganancias[repe, :] = ganancias_ronda
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
    
    y_test_np = y_test.to_numpy()  # <-- si aún no lo convertiste
    
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



#-----------------------------------------------> evalua el modelo en test

def evaluar_en_test (df: pd.DataFrame, mejores_params:dict) -> tuple:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info(f"INICIANDO EVALUACIÓN EN TEST")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VAL]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VAL]

    
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
    logger.info(f"Período de test: {MES_TEST}")

    
    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]


    df_train_completo = convertir_clase_ternaria_a_target_polars(df_train_completo, baja_2_1=True) # Entreno el modelo con Baja+1 y Baja+2 == 1
    df_test = convertir_clase_ternaria_a_target_polars(df_test, baja_2_1=False) # valido la ganancia solamente con Baja+2 == 1

    X_train_completo = df_train_completo.drop(columns = ['clase_ternaria'])
    y_train_completo = df_train_completo['clase_ternaria']

    X_test = df_test.drop(columns = ['clase_ternaria'])
    y_test = df_test['clase_ternaria']

    
    train_data = lgb.Dataset(X_train_completo, label=y_train_completo)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Copiar los parámetros para no modificar el dict original
    mejores_params = mejores_params.copy()

    # Tomar la iteración óptima si existe
    num_boost_round = mejores_params.pop('best_iteration', None)
    if num_boost_round is None:
        num_boost_round = mejores_params.pop('num_boost_round', 200)  # fallback


    # Entrenar modelo con mejores parámetros
    model = lgb.train(mejores_params, 
                      train_data,
                      num_boost_round=num_boost_round,
                      feval=ganancia_ordenada
                    )   


    # Predecir probabilidades y binarizar
    y_pred_prob = model.predict(X_test)
    y_pred_binary = (y_pred_prob > 0.025).astype(int) 
  
    # Calcular solo la ganancia
    ganancia_test = calcular_ganancia(y_test, y_pred_binary)
  
    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
    verdaderos_positivos = np.sum((y_pred_binary == 1) & (y_test == 1))
    falsos_positivos = np.sum((y_pred_binary == 1) & (y_test == 0))
    verdaderos_negativos = np.sum((y_pred_binary == 0) & (y_test == 0))
    falsos_negativos = np.sum((y_pred_binary == 0) & (y_test == 1))

    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos + 1e-10)  # para evitar división por cero
    recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos + 1e-10)
    accuracy = (verdaderos_positivos + verdaderos_negativos) / total_predicciones

    resultados_test = {
        'ganancia_test': float(ganancia_test),
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
    graficar_importances_test(model)


    return resultados_test, y_pred_binary, y_test, y_pred_prob












































