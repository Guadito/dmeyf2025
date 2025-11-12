import os
import json
import logging
from .config import STUDY_NAME
from .config import BUCKET_NAME

logger = logging.getLogger(__name__)



def cargar_mejores_hiperparametros(archivo_base: str = None, n_top: int = 1) -> list[dict] | dict:
    """
        Carga los mejores hiperparámetros desde el archivo JSON de iteraciones de Optuna.
        Permite devolver solo el mejor o los top N trials.

        Args:
            archivo_base (str, opcional): Nombre base del archivo. Si es None, usa STUDY_NAME.
            n_top (int, opcional): Cantidad de mejores trials a devolver. 
                Si es 1, devuelve un dict. Si es >1, devuelve una lista de dicts.

        Returns:
            dict | list[dict]: 
        """
    
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    # Ruta del archivo dentro del bucket
    resultados_dir = os.path.join(BUCKET_NAME, "resultados")
    archivo = os.path.join(resultados_dir, f"{archivo_base}_iteraciones.json")
  
    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
  
        if not iteraciones:
            raise ValueError("No se encontraron iteraciones en el archivo")
  
        # Encontrar el top con mayor ganancia
        iteraciones_ordenadas = sorted(iteraciones, key=lambda x: x['value'], reverse=True)
        top_trials = iteraciones_ordenadas[:n_top]

        # Construir lista de hiperparámetros
        top_params_list = [t['params'] for t in top_trials]


    # Logs informativos
        for i, t in enumerate(top_trials, start=1):
            valor = t['value']
            trial = t.get('trial_number', 'N/A')
            logger.info(f"Top {i}: trial {trial}, valor {valor:,.0f}")
        
        logger.info(f"Archivo cargado: {archivo} con top {n_top} mejores trials")

        # Retorno según n_top
        if n_top == 1:
            return top_params_list[0]
        else:
            return top_params_list
    
    except FileNotFoundError:
            logger.error(f"No se encontró el archivo {archivo}")
            logger.error("Asegúrate de haber ejecutado la optimización con Optuna primero")
            raise
    except Exception as e:
            logger.error(f"Error al cargar mejores hiperparámetros: {e}")
            raise


#--------------------> modificación para log

def cargar_mejores_hiperparametros_completo(archivo_base: str = None, n_top: int = 1) -> list[dict] | dict:
    """
    Carga los mejores hiperparámetros desde el archivo JSON de iteraciones de Optuna.
    Convierte los parámetros optimizados en log-space a sus valores reales.
    Incluye num_boost_round promedio y original si existen.
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    resultados_dir = os.path.join(BUCKET_NAME, "resultados")
    archivo = os.path.join(resultados_dir, f"{archivo_base}_iteraciones.json")
  
    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
  
        if not iteraciones:
            raise ValueError("No se encontraron iteraciones en el archivo")
  
        # Ordenar por valor de ganancia
        iteraciones_ordenadas = sorted(iteraciones, key=lambda x: x['value'], reverse=True)
        top_trials = iteraciones_ordenadas[:n_top]
        top_params_list = []
        
        for i, t in enumerate(top_trials, start=1):
            params = t['params'].copy()
            
            # Mostrar parámetros originales
            logger.info(f"=== Trial {i} - Parámetros ORIGINALES ===")
            logger.info(f"{json.dumps(params, indent=2)}")
            
            # Reconversión de parámetros log-space a valores normales
            if 'num_leaves_exp' in params:
                params['num_leaves'] = int(round(2 ** params.pop('num_leaves_exp')))
            if 'min_child_samples_exp' in params:
                params['min_child_samples'] = int(round(2 ** params.pop('min_child_samples_exp')))
            if 'num_boost_round_exp' in params:
                params['num_boost_round_original'] = int(round(2 ** params.pop('num_boost_round_exp')))
            
            if 'num_boost_round' in t:
                params['num_boost_round'] = t['num_boost_round']
            
            # Mostrar parámetros reconvertidos
            logger.info(f"=== Trial {i} - Parámetros RECONVERTIDOS ===")
            logger.info(f"{json.dumps(params, indent=2)}")
            
            top_params_list.append(params)
            logger.info(f"Top {i}: trial {t.get('trial_number', 'N/A')}, ganancia {t['value']:,.0f}")
        
        logger.info(f"Archivo cargado: {archivo} con top {n_top} mejores trials")
        if n_top == 1:
            return top_params_list[0]
        else:
            return top_params_list
    
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo {archivo}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar mejores hiperparámetros: {e}")
        raise



# -------------------------------> estadísticas optuna

def obtener_estadisticas_optuna(archivo_base=None):
    """
    Obtiene estadísticas de la optimización de Optuna.
  
    Args:
        archivo_base: Nombre base del archivo
  
    Returns:
        dict: Estadísticas de la optimización
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    # Ruta del archivo dentro del bucket
    resultados_dir = os.path.join(BUCKET_NAME, "resultados")
    archivo = os.path.join(resultados_dir, f"{archivo_base}_iteraciones.json")
  
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    try:
        with open(archivo, 'r') as f:
            iteraciones = json.load(f)
  
        ganancias = [iter['value'] for iter in iteraciones]
  
        estadisticas = {
            'total_trials': len(iteraciones),
            'mejor_ganancia': max(ganancias),
            'peor_ganancia': min(ganancias),
            'ganancia_promedio': sum(ganancias) / len(ganancias),
            'top_5_trials': sorted(iteraciones, key=lambda x: x['value'], reverse=True)[:5]
        }
  
        logger.info("Estadísticas de optimización:")
        logger.info(f"  Total trials: {estadisticas['total_trials']}")
        logger.info(f"  Mejor ganancia: {estadisticas['mejor_ganancia']:,.0f}")
        logger.info(f"  Ganancia promedio: {estadisticas['ganancia_promedio']:,.0f}")
  
        return estadisticas
  
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        raise