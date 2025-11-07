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

# ---------------------------> wilcoxon

def evaluar_wilcoxon(df: pd.DataFrame, top_params: list, n_seeds: int = 10) -> dict:
    """
    Evalúa los parámetros de los mejores modelos con n_seeds, calculando la ganancia por seed.
    Realiza pruebas de Wilcoxon pareadas entre todos los modelos y crea un ranking.

    Args
    ----
    df : pd.DataFrame
    top_params : list
        Lista de diccionarios con hiperparámetros a evaluar.
    n_seeds : int, optional
        Número de semillas. Por defecto 10.

    Returns
    -------
    dict con las siguientes claves:
        'mejor_modelo' : int
            Índice en top_params del modelo ganador (según ranking).
        'mejor_params' : dict
            Hiperparámetros del modelo ganador.
        'ranking' : list of tuples
            Lista ordenada [(idx, n_victorias, mediana_ganancia), ...] ordenada por victorias y mediana.
        'ganancias_por_seed' : list of lists
            Ganancias por seed para cada modelo: [[g1_seed1, ...], [g2_seed1, ...], ...].
        'wilcoxon_pvals' : dict
            Diccionario con p-values por par: keys = (i, j) -> p-value (i<j).
    """

    if len(top_params) < 2:
        logger.warning("Se necesitan al menos 2 modelos para comparar con Wilcoxon.")
        return {
            'mejor_modelo': 0 if top_params else None,
            'mejor_params': top_params[0] if top_params else None,
            'ranking': [],
            'ganancias_por_seed': [],
            'wilcoxon_pvals': {}
        }

    logger.info(f"Evaluando {len(top_params)} modelos con {n_seeds} semillas cada uno...")


    df_train = df[df['foto_mes'].isin(MES_TRAIN)]
    df_val = df[df['foto_mes'] == MES_VAL]


    df_train = convertir_clase_ternaria_a_target_polars(df_train, baja_2_1=True) # Entreno el modelo con Baja+1 y Baja+2 == 1
    df_val = convertir_clase_ternaria_a_target_polars(df_val, baja_2_1=False) # valido la ganancia solamente con Baja+2 == 1

    
    X_train = df_train.drop(columns=['clase_ternaria'])
    y_train = df_train['clase_ternaria']
    X_val = df_val.drop(columns=['clase_ternaria'])
    y_val = df_val['clase_ternaria']


    ganancias_top = []
    pvalues_dict = {}
 
    
    for idx, params in enumerate(top_params):
        ganancias = []
        logger.info(f"Modelo {idx + 1}/{len(top_params)}...")

        for seed in range(n_seeds):
            params_copy = params.copy()
            params_copy['seed'] = seed
            #Se toma la cantidad óptima de iteraciones
            num_boost_round = params_copy.pop('best_iteration', None)
            if num_boost_round is None:
                num_boost_round = params_copy.pop('num_boost_round', 200)


            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params_copy, 
                train_data,
                num_boost_round=num_boost_round, 
                feval=ganancia_ordenada,
                callbacks=[lgb.log_evaluation(0)]
            )

            y_pred_prob = model.predict(X_val)
            y_pred_binary = (y_pred_prob > 0.025).astype(int) 
            g = calcular_ganancia(y_val, y_pred_binary) 
            ganancias.append(g)

        ganancias_top.append(ganancias)
        logger.info(f"  Modelo {idx}: Mediana {np.median(ganancias):,.0f}")
    


    # Comparaciones Wilcoxon
    logger.info("\nComparaciones Wilcoxon:")
    n_modelos = len(ganancias_top)
    victorias = [0] * n_modelos
    
    for i, j in combinations(range(n_modelos), 2):
        try:
            _, p = wilcoxon(ganancias_top[i], ganancias_top[j])
        except ValueError:
            p = 1.0  # si no se puede comparar (mismas ganancias o longitudes)
        pvalues_dict[(i, j)] = p
        
        if p < 0.05:
            ganador = i if np.median(ganancias_top[i]) > np.median(ganancias_top[j]) else j
            victorias[ganador] += 1
            logger.info(f"  Modelo {ganador} > Modelo {i if ganador==j else j} (p={p:.3f})")
    
    # --- Ranking final ---
    ranking = [(idx, victorias[idx], np.median(ganancias_top[idx])) for idx in range(n_modelos)]
    ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)

    logger.info("\nRanking final:")
    for rank, (idx, vict, med) in enumerate(ranking, 1):
        logger.info(f"  #{rank} Modelo {idx}: {vict} victorias | Mediana {med:,.0f}")

    mejor_modelo = ranking[0][0]
    logger.info(f"Mejor modelo: {mejor_modelo}")

    
    return {
        'mejor_modelo': mejor_modelo,
        'mejor_params': top_params[mejor_modelo],
        'ranking': ranking,
        'ganancias_por_seed': ganancias_top,
        'wilcoxon_pvals': pvalues_dict
    }
