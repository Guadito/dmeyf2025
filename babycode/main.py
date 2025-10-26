import pandas as pd
import datetime
import os
import sys
import logging
from sklearn.model_selection import train_test_split

from src.features import *
from src.config import  *
from src.loader import *
from src.optimization import *
from src.best_params import *
from src.grafico_test import *
from src.final_training import *
from src.kaggle import *
import gc



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
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_VAL: {MES_VAL}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"TRAIN_FINAL: {FINAL_TRAIN}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")



def main():
    logger.info("Inicio de ejecucion.")
    
    # 1- cargar datos 
    #os.makedirs(BUCKET_NAME, exist_ok=True)
    
    
    # Y realizar FE  
    df_f = cargar_datos(DATA_PATH_BASE_VM)
    df_f = crear_clase_ternaria(df_f)
    #df_f = realizar_feature_engineering(df_f, lags = 3)

    #  #SAMPLE
    #n_sample = 50000
    #df_f, _ = train_test_split(
    #    df_f,
    #    train_size=n_sample,
    #    stratify=df_f['clase_ternaria'],
    #    random_state=42)

    col_montos = select_col_montos(df_f)
    df_f = feature_engineering_rank_pos_batch(df_f, col_montos)
    col = [c for c in df_f.columns if c not in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
    df_f = feature_engineering_lag_delta_polars(df_f, col, cant_lag = 2)
    col = [c for c in df_f.columns if c not in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
    #df_f = feature_engineering_rolling_mean(df_f, col, ventana = 3)
    #df_f.to_csv(DATA_PATH_TRANS_VM)


    #Con FE realizado
    #df_f = cargar_datos(DATA_PATH_TRANS_VM)

    # 2 - optimización de hiperparámetros
    logger.info("=== INICIANDO OPTIMIZACIÓN DE HIPERPARAMETROS ===")
    study = optimizar(df_f, n_trials= 50)  

    # 3 - Aplicar wilcoxon para obtener el modelo más significativo
    logger.info("=== APLICACIÓN TEST DE WILCOXON ===")  
    best_params = cargar_mejores_hiperparametros(n_top = 5)
    resultado = evaluar_wilcoxon(df_f, best_params, n_seeds = 10)
    

    # 4 - Evaluar modelo en test
    params_best_model = resultado['mejor_params']
    resultados_test, y_pred_binary, y_test, y_pred_prob = evaluar_en_test(df_f, params_best_model)

    
    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    logger.info(f"Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")

    
    # Grafico de test
    logger.info("=== GRAFICO DE TEST ===")
    ruta_grafico_avanzado = crear_grafico_ganancia_avanzado(y_true=y_test, y_pred_proba=y_pred_prob)
    logger.info(f"Gráficos generados: {ruta_grafico_avanzado}")


    logger.info("=== GENERANDO TABLA DE DECISIÓN DE CORTE ===")
    

    cortes = [9000, 9500, 10000, 10500]

    df_resultados = simular_cortes_kaggle(
    y_pred_prob=y_pred_prob,
    y_test=y_test,
    cortes=cortes,
    ganancia_por_corte=ganancia_por_corte,
    random_state=42)

    # Resume las ganancias promedio por corte
    df_resumen = resumen_cortes(df_resultados)
    print("\n=== RESULTADOS DE SIMULACIÓN DE CORTES ===")
    print(df_resumen.to_string(index=False))

    
    # 7 Entrenar modelo final
    logger.info("=== ENTRENAMIENTO FINAL ===")
    logger.info("Preparar datos para entrenamiento final")
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_f)

    # Entrenar modelo final
    logger.info("Entrenar modelo final")
    modelo_final = entrenar_modelo_final(X_train, y_train, params_best_model)

    # Generar predicciones finales
    logger.info("Generar predicciones finales")

    generar_predicciones_finales_por_umbral(modelo_final, X_predict, clientes_predict, umbrales=[0.020, 0.025, 0.029, 0.032])
    generar_predicciones_por_cantidad(modelo_final, X_predict, clientes_predict, cantidades = [9000, 9500, 10000, 10500, 12000, 12500, 13000])

    # 4 Guardar el DataFrame resultante
    #path = "Data/competencia_01_lag.csv"
    #df.to_csv(path, index=False)
    #logger.info(f"DataFrame resultante guardado en {path}")
    
    logger.info(f">>>> Ejecución finalizada <<<< Para más detalles, ver {nombre_log}")




if __name__ == "__main__":
    main()