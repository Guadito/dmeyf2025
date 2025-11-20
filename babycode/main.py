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

fecha = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
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
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"TRAIN_FINAL: {FINAL_TRAIN}")
logger.info(f"FINAL_PREDICT: {FINAL_PREDICT}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")



def main():
    logger.info("Inicio de ejecucion.")
    
    # 1- cargar datos 
    df_f = cargar_datos(DATA_PATH_BASE_VM)
    df_f = crear_clase_ternaria(df_f)    

    #SAMPLE
    #n_sample = 100000
    #df_f, _ = train_test_split(
    #    df_f,
    #    train_size=n_sample,
    #    stratify=df_f['clase_ternaria'],
    #    random_state=42)


    cols_to_drop = ['mprestamos_personales', 'cprestamos_personales'] # 'visa_cadelantosefectivo' 'cdescubierto_preacordado'
    df_f = drop_columns(df_f, cols_to_drop)
    logger.info(f"✓ Datos eliminando prestamos personales: {df_f.shape}")
    
    df_f = df_f.with_columns([(pl.col("foto_mes") % 100).alias("nmes")])
    logger.info(f"✓ Datos agregando mes as num: {df_f.shape}")
    
    df_f = normalizar_ctrx_quarter(df_f)
    logger.info(f"✓ Datos luego de normalización ctrx_quarter: {df_f.shape}")

    #col_montos = select_col_montos(df_f)
    col = ['mpayroll']
    df_f = generar_sobre_edad(df_f, col_montos)
    logger.info(f"✓ Datos agregando quarter/edad: {df_f.shape}")

    col = [c for c in df_f.columns if c not in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
    df_f = feature_engineering_lag_delta_polars(df_f, col, cant_lag = 2)
    cols_to_drop = ['periodo0']
    df_f = drop_columns(df_f, cols_to_drop)
    logger.info(f"✓ Datos luego de agregar lags: {df_f.shape}")
    
    df_df = tendencia_polars(df_f, col, ventana=6, tendencia=True, minimo=False, maximo=False, promedio=False)
    logger.info(f"✓ Datos luego de agregar tendencias: {df_f.shape}")
    
    #df_f = zero_replace(df_f)
    
    #col_montos = select_col_montos(df_f)
    #df_f = feature_engineering_rank_neg_batch(df_f, col_montos)
    

    

    
    #2 - entrenar el modelo y evaluar ganancias
    training = (eliminar_meses_lista(MES_TRAIN, mes_inicio=202001, mes_fin=202006)
    validation = MES_TEST
    lgb_train, X_train, y_train, X_val, y_val = preparar_datos_training_lgb(df_f, 
                                                                                training=training, 
                                                                                validation=validation, 
                                                                                undersampling_0= 0.2,
                                                                                qcanaritos = 5)
    


    modelo = entrenar_modelo(lgb_train, PARAMETROS_LGBM_Z, tipo="intermedio")
    cortes = [9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500]
    _,_,_,_,_,_,mejor_corte = evaluar_en_test(modelo, X_val, y_val, cortes=cortes, corte_fijo= 11000)


    #3 - entrenar el modelo final y predecir
    training =  (eliminar_meses_lista(FINAL_TRAIN, mes_inicio=202001, mes_fin=202006)
    predict = FINAL_PREDICT
    lgb_train_final, X_train_final, y_train_final, X_pred, clientes_predict = preparar_datos_final_zlgb (df_f, 
                                                                                                         training=training, 
                                                                                                         predict=predict, 
                                                                                                         undersampling_0= 0.2,
                                                                                                         qcanaritos = 5)

    modelo_final = entrenar_modelo(lgb_train_final, PARAMETROS_LGBM_Z, tipo="final")
    generar_predicciones_por_cantidad(modelo_final, X_pred, clientes_predict, corte = mejor_corte)



    logger.info(f">>>> Ejecución finalizada <<<< Para más detalles, ver {nombre_log}")




if __name__ == "__main__":
    main()