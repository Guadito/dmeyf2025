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


# ----------------------------------- > CV

def objetivo_ganancia_cv(trial, df) -> float: 
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos

  
    Description:
    Función objetivo que maximiza ganancia con k-fold.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento con kfold y un porcentaje de la clase CONTINÚA
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar
    
    
    params = {
        'verbosity': -1,
        'random_state': SEMILLAS[0],
        'objective': 'binary',
        'metric': 'None',  
        'max_bin': PARAMETROS_LGBM['max_bin'], 
        'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGBM['learning_rate'][0], PARAMETROS_LGBM['learning_rate'][1]),
        'num_leaves': trial.suggest_int('num_leaves', PARAMETROS_LGBM['num_leaves'][0], PARAMETROS_LGBM['num_leaves'][1]),
        'max_depth': trial.suggest_int('max_depth', PARAMETROS_LGBM['max_depth'][0], PARAMETROS_LGBM['max_depth'][1]),
        'min_child_samples': trial.suggest_int('min_child_samples', PARAMETROS_LGBM['min_child_samples'][0], PARAMETROS_LGBM['min_child_samples'][1]),
        'subsample': trial.suggest_float('subsample', PARAMETROS_LGBM['subsample'][0], PARAMETROS_LGBM['subsample'][1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', PARAMETROS_LGBM['colsample_bytree'][0], PARAMETROS_LGBM['colsample_bytree'][1]),
        'min_split_gain': trial.suggest_float('min_split_gain', PARAMETROS_LGBM['min_split_gain'][0], PARAMETROS_LGBM['min_split_gain'][1]),
        'zero_as_missing': trial.suggest_categorical('zero_as_missing', [True, False]) 
        }
    
    num_boost_round = trial.suggest_int('num_boost_round', PARAMETROS_LGBM['num_boost_round'][0], PARAMETROS_LGBM['num_boost_round'][1])
    undersampling_ratio = PARAMETROS_LGBM['undersampling']


    # Preparar datos de entrenamiento (TRAIN  VALIDACION)
    if isinstance(GENERAL_TRAIN, list):
        train_data = df[df['foto_mes'].isin(GENERAL_TRAIN)]
    else: train_data = df[df['foto_mes'] == GENERAL_TRAIN]

    train_data = convertir_clase_ternaria_a_target(train_data, baja_2_1=True) # Entreno el modelo con Baja+1 y Baja+2 == 1

    logger.info(
    f"Dataset GENERAL TRAIN - Antes del subsampleo de la clase CONTINUA: "
    f"Clase 1: {len(train_data[train_data['clase_ternaria'] == 1])}, "
    f"Clase 0: {len(train_data[train_data['clase_ternaria'] == 0])}"
    )

    # SUBMUESTREO
    clase_1 = train_data[train_data['clase_ternaria'] == 1]
    clase_0 = train_data[train_data['clase_ternaria'] == 0]    
    
    semilla = SEMILLAS[0] if isinstance(SEMILLAS, list) else SEMILLAS
    clase_0_sample = clase_0.sample(frac=undersampling_ratio, random_state=semilla)
    train_data = pd.concat([clase_1, clase_0_sample], axis=0).sample(frac=1, random_state=semilla)
    

    X_train = train_data.drop(columns = ['clase_ternaria'])
    y_train = train_data['clase_ternaria']
    

    lgb_train = lgb.Dataset(X_train, label=y_train)
    

    cv_results = lgb.cv(params,
                    num_boost_round=num_boost_round,
                    nfold=5,
                    stratified=True,
                    train_set=lgb_train,
                    shuffle = True,  
                    seed = SEMILLAS[0],
                    feval=ganancia_ordenada,   #METRIC SE LA DECLARA VACÍA Y EN SU LUGAR SE USA FEVAL
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)])
    
    # Predecir probabilidades y binarizar
    ganancias_cv = cv_results['valid ganancia-mean']
    ganancia_maxima = np.max(ganancias_cv)
    best_iter = np.argmax(ganancias_cv)

    #Guardar el nro original de árboles y el optimizado:
    num_boost_round_original = trial.params['num_boost_round']
    trial.set_user_attr('num_boost_round_original', num_boost_round_original)
    trial.set_user_attr('best_iteration', int(best_iter)) 
    trial.params['num_boost_round'] = int(best_iter)

    
    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_maxima:,.0f}")
    logger.debug(f"Trial {trial.number}: Mejor iteracion = {best_iter:,.0f}")
    
    guardar_iteracion_cv(trial, ganancia_maxima, ganancias_cv, archivo_base=None)
    
    return ganancia_maxima


#---------------------------------------------------------------> Parametrización OPTUNA  aplicación de OB

def optimizar_cv(df, n_trials=int, study_name: str = None ) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
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


    logger.info(f"Iniciando optimización con CV {n_trials} trials")
    logger.info(f"Configuración: TRAIN para CV={GENERAL_TRAIN}, SEMILLA={SEMILLAS[0]}")
  
        # Crear estudio de Optuna
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler = optuna.samplers.TPESampler(seed= SEMILLAS[0]), 
        storage=storage,
        load_if_exists=True
    )

    # Aquí iría tu función objetivo y la optimización
    study.optimize(lambda trial: objetivo_ganancia_cv(trial, df), n_trials=n_trials)

    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Optimizacion CV completada. Mejor ganancia promedio: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
    logger.info(f"Total trials: {len(study.trials)}")

    return study


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



