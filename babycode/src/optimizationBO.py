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

def objetivo_ganancia_semillerio(trial, df, undersampling: int = 1, repeticiones: int = 1, ksemillerio: int = 1 ) -> float: 
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos

  
    Description:
    Función objetivo que maximiza ganancia 
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación a partir de yaml
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """

    learning_rate = trial.suggest_float('learning_rate', PARAMETROS_LGBM['learning_rate'][0],PARAMETROS_LGBM['learning_rate'][1],log=True) 
    num_leaves_exp = trial.suggest_float('num_leaves_exp', np.log2(PARAMETROS_LGBM['num_leaves'][0]), np.log2(PARAMETROS_LGBM['num_leaves'][1]))
    num_leaves = int(round(2 ** num_leaves_exp))
    max_depth = trial.suggest_int('max_depth', PARAMETROS_LGBM['max_depth'][0],PARAMETROS_LGBM['max_depth'][1])
    
    # RESTRICCIÓN: num_leaves debe ser <= 2^max_depth  Si no se cumple, pruning
    if num_leaves > 2 ** max_depth:
        logger.warning(f"Trial {trial.number} PRUNED: num_leaves ({num_leaves}) > 2^max_depth ({2**max_depth})")
        raise optuna.exceptions.TrialPruned()
    
    min_child_samples_exp = trial.suggest_float('min_child_samples_exp',
                                                np.log2(PARAMETROS_LGBM['min_child_samples'][0]), 
                                                np.log2(PARAMETROS_LGBM['min_child_samples'][1]))
    min_child_samples = int(round(2 ** min_child_samples_exp))
    
    n_train = len(df[df['foto_mes'].isin(MES_TRAIN) if isinstance(MES_TRAIN, list) else df['foto_mes'] == MES_TRAIN])
    if min_child_samples * num_leaves > n_train:
        logger.warning(f"Trial {trial.number} PRUNED: min_child_samples*num_leaves ({min_child_samples * num_leaves}) > n_train ({n_train})")
        raise optuna.exceptions.TrialPruned()

    subsample = trial.suggest_float('subsample', PARAMETROS_LGBM['subsample'][0], 
                                    PARAMETROS_LGBM['subsample'][1])   
    
    colsample_bytree = trial.suggest_float('colsample_bytree', 
                                           PARAMETROS_LGBM['colsample_bytree'][0], 
                                           PARAMETROS_LGBM['colsample_bytree'][1])

    min_split_gain = trial.suggest_float('min_split_gain', 
                                         PARAMETROS_LGBM['min_split_gain'][0], 
                                         PARAMETROS_LGBM['min_split_gain'][1])

    num_boost_round_exp = trial.suggest_float('num_boost_round_exp',
                                              np.log2(PARAMETROS_LGBM['num_boost_round'][0]),
                                              np.log2(PARAMETROS_LGBM['num_boost_round'][1]))
    num_boost_round = int(round(2 ** num_boost_round_exp))


    logger.info(f"Trial {trial.number} - Hiperparámetros: lr={learning_rate:.6f}, "
                f"leaves={num_leaves}, depth={max_depth}, min_child={min_child_samples}, "
                f"subsample={subsample:.3f}, colsample={colsample_bytree:.3f}, "
                f"rounds={num_boost_round}")
    
    # Hiperparámetros
    params = {
        'verbosity': -1,
        'metric': 'None',
        'objective': 'binary',
        'max_bin': PARAMETROS_LGBM['max_bin'], 
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_split_gain': min_split_gain,
        }
    
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].isin(MES_TRAIN)]
    else:
        df_train = df[df['foto_mes'] == MES_TRAIN]
    df_val = df[df['foto_mes'] == MES_VAL]

    
    logger.info(
        f"Tamaño train: {len(df_train)}. "
        f"Rango train: {min(MES_TRAIN) if isinstance(MES_TRAIN, list) else MES_TRAIN} - "
        f"{max(MES_TRAIN) if isinstance(MES_TRAIN, list) else MES_TRAIN}. "
        f"Tamaño val: {len(df_val)}. Val: {MES_VAL}")


    #Convierto a binaria la clase ternaria 
    df_train = convertir_clase_ternaria_a_target_polars(df_train, baja_2_1=True) # Entreno el modelo con Baja+1 y Baja+2 == 1
    df_val = convertir_clase_ternaria_a_target_polars(df_val, baja_2_1=False) # valido la ganancia solamente con Baja+2 == 1

    #Subsampleo
    df_train = aplicar_undersampling_clase0(df_train, undersampling, seed= SEMILLAS[0])   #VER SEMILLA.

    df_train['clase_ternaria'] = df_train['clase_ternaria'].astype(np.int8)
    df_val['clase_ternaria'] = df_val['clase_ternaria'].astype(np.int8)

    X_train = df_train.drop(columns = ['clase_ternaria'])
    y_train = df_train['clase_ternaria']
    lgb_train = lgb.Dataset(X_train, label=y_train)

    X_val = df_val.drop(columns = ['clase_ternaria'])
    y_val = df_val['clase_ternaria']
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    logger.info(f"Trial {trial.number} - Configuración semillerio: {repeticiones} repeticiones x {ksemillerio} semillas = {repeticiones * ksemillerio} modelos totales")

    rng = np.random.default_rng(SEMILLAS[0])
    semillas_totales = rng.choice(np.arange(100_000, 1_000_000), 
                              size=ksemillerio * repeticiones, 
                              replace=False)
    
    lista_ganancias_repeticion = []
    lista_best_iters_total = []

    for repe in range (repeticiones): 
        desde = repe * ksemillerio
        hasta = (repe + 1) * ksemillerio
        semillas_ronda = semillas_totales[desde:hasta]

        y_pred_acum = np.zeros(len(X_val))
        
        lista_best_iters_ronda = []


        logger.info("="*80)
        logger.info(f"🔁 [Trial {trial.number}] Repetición {repe+1}/{repeticiones}")
        logger.info(f"🌱 Semillas en esta ronda: {list(semillas_ronda)}")
        logger.info("="*80)


        for i, semilla in enumerate(semillas_ronda, start=1):    
            params['random_state'] = semilla

            model = lgb.train(params,
                            num_boost_round=num_boost_round,              
                            train_set=lgb_train,
                            valid_sets=[lgb_val],  
                            feval=ganancia_ordenada,  
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)])
        
            # Obtener mejor iteracion 
            best_iter_semilla= model.best_iteration        #Mejor iteración de la semilla
            lista_best_iters_ronda.append(best_iter_semilla)  #Guarda las mejores iteraciones de la repetición
            lista_best_iters_total.append(best_iter_semilla) #Guarda las mejores iteraciones del trial


            y_pred_proba_i = model.predict(X_val, num_iteration=best_iter_semilla)
            y_pred_acum += y_pred_proba_i

        
        y_pred_prom = y_pred_acum / len(semillas_ronda)
        _, ganancia_ronda, _ = ganancia_ordenada_meseta(y_pred_prom, y_val)
        lista_ganancias_repeticion.append(ganancia_ronda)

            # Log detallado de la repetición
        logger.info(f"[Trial {trial.number}] Repetición {repe+1} completada:")
        logger.info(f"Ganancia meseta: {ganancia_ronda:,.0f}")

    ganancia_total_promedio = np.mean(lista_ganancias_repeticion)
    ganancia_sd = np.std(lista_ganancias_repeticion)

    logger.info("FINAL DEL TRIAL")
    logger.info(f"Trial {trial.number} - Promedio de ganancias (meseta): {ganancia_total_promedio:,.0f} ± {ganancia_sd:,.0f}")
    logger.info(f"Iteraciones totales: {lista_best_iters_total}")
    logger.info("="*80)

    # Guardar información del trial
    best_iter_promedio = int(np.mean(lista_best_iters_total))
    logger.info(f"Best iter promedio para trial {trial.number}: {best_iter_promedio}")

    num_boost_round_original = int(round(2 ** trial.params['num_boost_round_exp']))
    trial.set_user_attr('num_boost_round_original', num_boost_round_original)
    trial.set_user_attr('best_iteration', int(best_iter_promedio))
    trial.params['num_boost_round'] = best_iter_promedio #actualizar el num_boost_round
        
    guardar_iteracion(trial, ganancia_total_promedio, archivo_base=None)
    
    return ganancia_total_promedio


#---------------------------------------------------------------> func crear o cargar estudio OPTUNA

def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente desde SQLite.
  
    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad
  
    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    study_name = STUDY_NAME
  
    if semilla is None:
        semilla = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA
  
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "optuna_db")
    os.makedirs(path_db, exist_ok=True)
  
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"
  
    # Verificar si existe un estudio previo
    if os.path.exists(db_file):
        logger.info(f"⚡ Base de datos encontrada: {db_file}")
        logger.info(f"🔄 Cargando estudio existente: {study_name}")
  
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials_previos = len(study.trials)
  
            logger.info(f"✅ Estudio cargado exitosamente")
            logger.info(f"📊 Trials previos: {n_trials_previos}")
  
            if n_trials_previos > 0:
                logger.info(f"🏆 Mejor ganancia hasta ahora: {study.best_value:,.0f}")
  
            return study
  
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el estudio: {e}")
            logger.info(f"🆕 Creando nuevo estudio...")
    else:
        logger.info(f"🆕 No se encontró base de datos previa")
        logger.info(f"📁 Creando nueva base de datos: {db_file}")
  
    # Crear nuevo estudio
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler = optuna.samplers.TPESampler(seed= SEMILLAS[0]), 
        storage=storage,
        load_if_exists=True
    )
  
    logger.info(f"✅ Nuevo estudio creado: {study_name}")
    logger.info(f"💾 Storage: {storage}")
  
    return study



#---------------------------------------------------------------> Aplicación de OB

def optimizar(df: pd.DataFrame, n_trials: int, study_name: str = None, 
              undersampling: float = 0.01, repeticiones: int = 1, ksemillerio: int = 1) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
        undersampling: Undersampling para entrenamiento
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VAL}, SEMILLA={SEMILLAS[0]}")
  
    # Crear o cargar estudio desde DuckDB
    study = crear_o_cargar_estudio(study_name, SEMILLAS[0])

    # Calcular cuántos trials faltan
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
  
    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}")
        logger.info(f"📝 Trials a ejecutar: {trials_a_ejecutar} (total objetivo: {n_trials})")
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")
  
    # Ejecutar optimización
    if trials_a_ejecutar > 0:
        study.optimize(lambda trial: objetivo_ganancia_semillerio(trial, df, undersampling=undersampling), n_trials=trials_a_ejecutar)
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
  
    return study   

# --------------- > evaluar modelo con semillerio

def evaluar_en_test_semillerio (df: pd.DataFrame, mejores_params:dict) -> tuple:
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

    repeticiones = PARAMETROS_LGBM.get('REPETICIONES', 1)
    ksemillerio= PARAMETROS_LGBM.get('KSEMILLERIO', 1)

    logger.info(f"Trial {trial.number} - Configuración semillerio: {repeticiones} repeticiones x {ksemillerio} semillas = {repeticiones * ksemillerio} modelos totales")

    rng = np.random.default_rng(SEMILLAS[0])
    semillas_totales = rng.choice(np.arange(100_000, 1_000_000), 
                              size=ksemillerio * repeticiones, 
                              replace=False)

    for i, semilla in enumerate(semillas_ronda, start=1):    
            params['random_state'] = semilla
        
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












































#-----------------------------------------------------------  optim bayesiana simple

def objetivo_ganancia(trial, df) -> float: 
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos

  
    Description:
    Función objetivo que maximiza ganancia 
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación a partir de yaml
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """

    learning_rate = trial.suggest_float('learning_rate', PARAMETROS_LGBM['learning_rate'][0],PARAMETROS_LGBM['learning_rate'][1],log=True) 
    
    num_leaves_exp = trial.suggest_float('num_leaves_exp', np.log2(PARAMETROS_LGBM['num_leaves'][0]), np.log2(PARAMETROS_LGBM['num_leaves'][1]))
    num_leaves = int(round(2 ** num_leaves_exp))
    max_depth = trial.suggest_int('max_depth', PARAMETROS_LGBM['max_depth'][0],PARAMETROS_LGBM['max_depth'][1])
    
    # RESTRICCIÓN IMPORTANTE: num_leaves debe ser <= 2^max_depth
    # Si no se cumple, pruning
    if num_leaves > 2 ** max_depth:
        raise optuna.exceptions.TrialPruned()
    
    min_child_samples_exp = trial.suggest_float('min_child_samples_exp',np.log2(PARAMETROS_LGBM['min_child_samples'][0]), np.log2(PARAMETROS_LGBM['min_child_samples'][1]))
    min_child_samples = int(round(2 ** min_child_samples_exp))
    

    n_train = len(df[df['foto_mes'].isin(MES_TRAIN) if isinstance(MES_TRAIN, list) else df['foto_mes'] == MES_TRAIN])
    if min_child_samples * num_leaves > n_train:
        raise optuna.exceptions.TrialPruned()

    subsample = trial.suggest_float('subsample', PARAMETROS_LGBM['subsample'][0], PARAMETROS_LGBM['subsample'][1])    
    colsample_bytree = trial.suggest_float('colsample_bytree', PARAMETROS_LGBM['colsample_bytree'][0], PARAMETROS_LGBM['colsample_bytree'][1])

    min_split_gain = trial.suggest_float('min_split_gain', PARAMETROS_LGBM['min_split_gain'][0], PARAMETROS_LGBM['min_split_gain'][1])

    num_boost_round_exp = trial.suggest_float('num_boost_round_exp',np.log2(PARAMETROS_LGBM['num_boost_round'][0]),np.log2(PARAMETROS_LGBM['num_boost_round'][1]))
    num_boost_round = int(round(2 ** num_boost_round_exp))

    undersampling = PARAMETROS_LGBM['undersampling']

    
    # Hiperparámetros a optimizar
    params = {
        'verbosity': -1,
        'metric': 'None',
        'objective': 'binary',
        'random_state': SEMILLAS[0],
        'max_bin': PARAMETROS_LGBM['max_bin'], 
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_split_gain': min_split_gain,
        }
    


    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].isin(MES_TRAIN)]
    else:
        df_train = df[df['foto_mes'] == MES_TRAIN]
    df_val = df[df['foto_mes'] == MES_VAL]

    
    logger.info(
        f"Tamaño train: {len(df_train)}. "
        f"Rango train: {min(MES_TRAIN) if isinstance(MES_TRAIN, list) else MES_TRAIN} - "
        f"{max(MES_TRAIN) if isinstance(MES_TRAIN, list) else MES_TRAIN}. "
        f"Tamaño val: {len(df_val)}. Val: {MES_VAL}"
    )


    #Convierto a binaria la clase ternaria 
    df_train = convertir_clase_ternaria_a_target_polars(df_train, baja_2_1=True) # Entreno el modelo con Baja+1 y Baja+2 == 1
    df_val = convertir_clase_ternaria_a_target_polars(df_val, baja_2_1=False) # valido la ganancia solamente con Baja+2 == 1

    #Subsampleo
    df_train = aplicar_undersampling_clase0(df_train, undersampling)


    df_train['clase_ternaria'] = df_train['clase_ternaria'].astype(np.int8)
    df_val['clase_ternaria'] = df_val['clase_ternaria'].astype(np.int8)

    X_train = df_train.drop(columns = ['clase_ternaria'])
    y_train = df_train['clase_ternaria']
    lgb_train = lgb.Dataset(X_train, label=y_train)
    

    X_val = df_val.drop(columns = ['clase_ternaria'])
    y_val = df_val['clase_ternaria']
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    
    model = lgb.train(params,
                    num_boost_round=num_boost_round,              
                    train_set=lgb_train,
                    valid_sets=[lgb_val],  
                    feval=ganancia_ordenada,  
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)])
    

    # Obtener el mejor número de iteraciones (después del early stopping)
    best_iter = model.best_iteration

    #Predecir y calcular ganancia
    y_pred_proba = model.predict(X_val, num_iteration=best_iter)
    ganancia_ordenada_, ganancia_total, _ = ganancia_ordenada(y_pred_proba, lgb_val)

    # Guardar información del trial
    num_boost_round_original = int(round(2 ** trial.params['num_boost_round_exp']))
    trial.set_user_attr('num_boost_round_original', num_boost_round_original)
    trial.set_user_attr('best_iteration', int(best_iter))
    trial.params['num_boost_round'] = int(best_iter) #actualizar el num_boost_round

    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
    logger.info(f"Trial {trial.number}: Mejor iteración = {best_iter}")
    
    guardar_iteracion(trial, ganancia_total, archivo_base=None)
    
    return ganancia_total


