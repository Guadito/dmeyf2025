import os
import yaml
import logging


logger = logging.getLogger(__name__)    

#Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


try:
    with open(PATH_CONFIG, "r") as file:
        _cfgGeneral = yaml.safe_load(file)    # <-- Lee el YAML
        _cfg = _cfgGeneral["competencia02"]   
        PARAMETROS_LGBM = _cfgGeneral["params_lgb_2"] 


    #Configuración global del proyecto
    STUDY_NAME = _cfgGeneral.get("STUDY_NAME", None)
    DATA_PATH_BASE_NT = _cfg.get("DATA_PATH_BASE_NT", "C:\\Users\\Guada\\Desktop\\babycode\\datasets\\competencia_01_crudo.csv")
    DATA_PATH_BASE_VM = _cfg.get("DATA_PATH_BASE_VM", "../datasets/competencia_01_crudo.csv")
    DATA_PATH_TRANS_VM = _cfg.get("DATA_PATH_TRANS_VM", "../datasets/competencia_01_crudo.csv")
    BUCKET_NAME = _cfgGeneral.get("BUCKET_NAME", None)
    SEMILLAS = _cfg.get("SEMILLAS", [42])
    MES_TRAIN = _cfg.get("MES_TRAIN",[])
    MES_VAL = _cfg.get("MES_VAL", [])
    MES_TEST = _cfg.get("MES_TEST",[])
    GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
    COSTO_ESTIMULO =   _cfg.get("COSTO_ESTIMULO", None)
    GENERAL_TRAIN = _cfg.get("GENERAL_TRAIN", [])
    FINAL_TRAIN = _cfg.get("FINAL_TRAIN", [])
    FINAL_PREDICT = _cfg.get("FINAL_PREDICT", "")
    undersampling = PARAMETROS_LGBM.get("undersampling", [])

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise