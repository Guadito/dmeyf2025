import pandas as pd
import datetime
import os
import sys
import logging
from sklearn.model_selection import train_test_split

from src.features import *
from src.config import  *
from src.loader import *
from src.optimizationBO import *
from src.mid_training import *
from src.best_params import *
from src.grafico_test import *
from src.final_training import *
from src.kaggle import *
import gc
from pathlib import Path
import numpy as np
logging.getLogger().setLevel(logging.ERROR)


# Crear carpeta de logs dentro del bucket
log_dir = os.path.join(BUCKET_NAME, "log")
os.makedirs(log_dir, exist_ok=True)

fecha = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"
ruta_log = os.path.join(log_dir, nombre_log)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(ruta_log),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Iniciando programa de optimización con log fechado")


### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DT crudo DATA_PATH_BASE_VM: {DATA_PATH_BASE_VM}")
logger.info(f"DT transformado DATA_PATH_2: {DATA_PATH_TRANS_VM}")
logger.info(f"BUCKET_NAME: {BUCKET_NAME}")
logger.info(f"SEMILLAS: {SEMILLAS}")
logger.info(f"Meses de entrenamiento para bayesiana: {MES_TRAIN}")
logger.info(f"Meses de entrenamiento para testeo: {MES_TRAIN2}")
logger.info(f"MES_VAL: {MES_VAL}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"TRAIN_FINAL: {FINAL_TRAIN}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")



def main():
    logger.info("Inicio de ejecucion.")
    
    # 1- cargar datos 
    df_f = cargar_datos(DATA_PATH_BASE_VM)
    df_f = crear_clase_ternaria(df_f)    


    cols_to_drop = ['mprestamos_personales', 'cprestamos_personales']  #'active_quarter', 'cprestamos_prendarios','mprestamos_prendarios', 'mpayroll_2', 'mpayroll_2', 'visa_cadelantosefectivo' ,'ctrx_quarter' 'cdescubierto_preacordado'
    df_f = drop_columns(df_f, cols_to_drop)
    
    df_f = df_f.with_columns([(pl.col("foto_mes") % 100).alias("nmes")])
    df_f = normalizar_ctrx_quarter(df_f)
    
    col = ['mpayroll']
    df_f = generar_sobre_edad(df_f, col)

    col = [c for c in df_f.columns if c not in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
    df_f = feature_engineering_lag_delta_polars(df_f, col, cant_lag = 2)
    cols_to_drop = ['periodo0']
    df_f = drop_columns(df_f, cols_to_drop)
    
    df_df = tendencia_polars(df_f, cols, ventana=6, tendencia=True, minimo=False, maximo=False, promedio=False)
    
    #df_f = zero_replace(df_f)
    
    #col_montos = select_col_montos(df_f)
    #df_f = feature_engineering_rank_neg_batch(df_f, col_montos)
    


    #2 - entrenar el modelo y evaluar ganancias
    training = [MES_TRAIN]
    validation = [MES_VAL]
    lgb_train, lgb_val, X_train, y_train, X_val, y_val = preparar_datos_training_lgb(df_f, training=training, validation=validation, undersampling_0= 0.05)


    modelo = entrenar_modelo(lgb_train, PARAMETROS_LGBM_Z)
    cortes = [9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500]
    _,_,_,_,_,_,mejor_corte = evaluar_en_test(modelo, X_val, y_val, cortes: list, corte_fijo: int = 11000):


    #3 - entrenar el modelo final y predecir
    
    training = FINAL_TRAIN
    predict = FINAL_PREDICT
    lgb_train_final, X_train_final, y_train_final, X_pred, clientes_predict = preparar_datos_final_zlgb (df_f, training=training, predict=predict, undersampling_0= 0.05)

    modelo_final = entrenar_modelo(lgb_train_final, PARAMETROS_LGBM_Z)
    generar_predicciones_finales(modelo_final, X_pred, clientes_predict, corte = mejor_corte)















    
    
    # 2 - optimización de hiperparámetros
    #logger.info("=== INICIANDO OPTIMIZACIÓN DE HIPERPARAMETROS ===")
    #study = optimizar(df_f, n_trials= 30, undersampling = 0.05, repeticiones = 1, ksemillerio = 7)  

    # 3 - Evaluar modelo en test
    #best_params = cargar_mejores_hiperparametros_completo(n_top = 1)
    #cortes = [9000, 9500, 10000, 10500, 12000, 12500, 13000, 16000, 18000]
    #resultados_test, resultados_df_fijo, resultados_df, y_pred_binary_mejor, y_test, y_pred_promedio_total = evaluar_en_test_semillerio(df_f, best_params, cortes, repeticiones = 1, ksemillerio = 100)

    # Resumen de evaluación en test
    #logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    #logger.info(f"Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    #logger.info(f"Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")

    # Grafico de test
    #logger.info("=== GRAFICO DE TEST ===")
    #ruta_grafico_avanzado = crear_grafico_ganancia_avanzado(y_true=y_test, y_pred_proba=y_pred_prob)
    #logger.info(f"Gráficos generados: {ruta_grafico_avanzado}")

    guardar_predicciones_finales(resultados_df_fijo)
    guardar_predicciones_finales(resultados_df)



    #logger.info("=== GENERANDO TABLA DE DECISIÓN DE CORTE ===")

    #cortes = [9000, 9500, 10000, 10500, 12000, 12500, 13000, 16000, 18000]

    #df_resultados = simular_cortes_kaggle(
    #y_pred_prob=y_pred_prob,
    #y_test=y_test,
    #cortes=cortes,
    #ganancia_por_corte=ganancia_por_corte,
    #random_state=42)

    # Resume las ganancias promedio por corte
    #df_resumen = resumen_cortes(df_resultados)
    #print("\n=== RESULTADOS DE SIMULACIÓN DE CORTES ===")
    #print(df_resumen.to_string(index=False))

    
    # 7 Entrenar modelo final
    #logger.info("=== ENTRENAMIENTO FINAL ===")
    #logger.info("Preparar datos para entrenamiento final")
    #X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_f)

    # Entrenar modelo final
    #logger.info("Entrenar modelo final")
    #modelo_final = entrenar_modelo_final(X_train, y_train, params_best_model)

    # Generar predicciones finales
    #logger.info("Generar predicciones finales")


    #generar_predicciones_por_cantidad(modelo_final, X_predict, clientes_predict, cantidades = [9000, 9500, 10000, 10500, 12000, 12500, 13000, 16000, 18000])

    # 4 Guardar el DataFrame resultante
    #path = "Data/competencia_01_lag.csv"
    #df.to_csv(path, index=False)
    #logger.info(f"DataFrame resultante guardado en {path}")
    
    logger.info(f">>>> Ejecución finalizada <<<< Para más detalles, ver {nombre_log}")




if __name__ == "__main__":
    main()