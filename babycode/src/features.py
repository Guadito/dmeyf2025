# src/features.py
import pandas as pd
import duckdb
import logging
import re
import sys
print(sys.executable)

import os
import duckdb
import pandas as pd 
import polars as pl
from .loader import crear_clase_ternaria 



logger = logging.getLogger("__name__")



def realizar_feature_engineering (df: pd.DataFrame, lags:int = 1) -> pd.DataFrame:

    df = crear_clase_ternaria(df)

    col_montos = select_col_montos(df)
    df = feature_engineering_rank_pos_batch(df, col_montos)

    col = [c for c in df.columns if c not in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
    df = feature_engineering_rolling_mean(df, col, ventana=3)
    df = feature_engineering_lag_delta_batch(df, col, cant_lag = lags)
    

    return df
    

#----------------------------> selecciona variables de montos 

def select_col_montos(df: pd.DataFrame) -> list:
    """
    Selecciona columnas de "montos" de un DataFrame según patrones:
      - columnas que empiezan con 'm'
      - columnas que contienen 'Master_m o Visa_m'
    Siempre excluye: 'numero_de_cliente' y 'foto_mes'
    
    Parámetros:
        df (pd.DataFrame): DataFrame original

    Retorna:
        list: columnas seleccionadas
    """
    # patrón: empieza con m  O contiene _m
    pattern_incl = re.compile(r'(^m|Master_m|Visa_m)')

    # columnas que cumplen el patrón
    selected_cols = [col for col in df.columns if pattern_incl.search(col)]

    # excluir columnas específicas
    excluded = {"numero_de_cliente", "foto_mes"}
    selected_cols = [col for col in selected_cols if col not in excluded]

    return selected_cols

#-----------------------------------------------------> Rank positivo batch

def feature_engineering_rank_pos_batch(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera rankings normalizados por signo para las columnas especificadas,
    procesando los datos por 'foto_mes' y columna para reducir el uso de memoria.
    """
    import duckdb

    if not columnas:
        raise ValueError("La lista de columnas no puede estar vacía")

    columnas_validas = [col for col in columnas if col in df.columns]
    if not columnas_validas:
        raise ValueError("No hay columnas válidas para rankear")

    logger.info(f"Generando ranking por batch: {len(columnas_validas)} columnas válidas.")
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads=1;")
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET memory_limit='2GB';")

    df_result = []

    for mes in sorted(df["foto_mes"].unique()):
        df_mes = df[df["foto_mes"] == mes].copy()
        logger.info(f"Procesando foto_mes={mes} con {len(df_mes)} filas...")

        con.register("df_temp", df_mes)

        for col in columnas_validas:
            sql = f"""
            SELECT
                {col},
                CASE
                    WHEN {col} = 0 THEN 0.0
                    ELSE PERCENT_RANK() OVER (
                        PARTITION BY SIGN({col})
                        ORDER BY {col}
                    )
                END AS rank_col
            FROM df_temp
            """
            df_rank = con.execute(sql).df()
            df_mes[col] = df_rank["rank_col"]

        con.unregister("df_temp")
        df_result.append(df_mes)

    con.close()
    resultado = pd.concat(df_result, ignore_index=True)
    logger.info(f"Feature engineering completado por batch y columna. Shape final: {resultado.shape}")
    return resultado





#-------------------------------> Crea LAG y DELTA en batch.
def feature_engineering_lag_delta_batch(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1, batch_size: int = 25) -> pd.DataFrame:
    """
    Genera variables de lag y delta para los atributos especificados utilizando SQL (DuckDB).
    Procesa columnas en batches para evitar problemas de memoria.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales.
    columnas : list[str]
        Lista de atributos para los cuales generar lags y deltas.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo.
    batch_size : int, default=25
        Número de columnas a procesar por batch.
    
    Returns
    -------
    pd.DataFrame
        DataFrame con las variables de lag y delta agregadas.
    """
    logger.info(f"Generando {cant_lag} lags y deltas para {len(columnas)} atributos en batches de {batch_size}")
    
    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags/deltas")
        return df
    
    # Validar columnas requeridas
    if 'numero_de_cliente' not in df.columns or 'foto_mes' not in df.columns:
        logger.error("El DataFrame debe contener 'numero_de_cliente' y 'foto_mes'")
        raise ValueError("Columnas requeridas no encontradas")
    
    # Iniciar con el DataFrame original
    df_result = df.copy()
    
    # Procesar columnas en batches
    num_batches = (len(columnas) - 1) // batch_size + 1
    
    for i in range(0, len(columnas), batch_size):
        batch_num = i // batch_size + 1
        batch_cols = columnas[i:i+batch_size]
        
        logger.info(f"Procesando batch {batch_num}/{num_batches}: {len(batch_cols)} columnas")
        
        # Construir la consulta SQL solo para este batch
        sql = "SELECT numero_de_cliente, foto_mes"
        
        for attr in batch_cols:
            if attr in df.columns:
                for j in range(1, cant_lag + 1):
                    sql += f', LAG("{attr}", {j}) OVER w AS "{attr}_lag_{j}"'
                    sql += f', ("{attr}" - LAG("{attr}", {j}) OVER w) AS "{attr}_delta_{j}"'
            else:
                logger.warning(f"El atributo {attr} no existe en el DataFrame")
        
        sql += " FROM df WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)"
        
        logger.debug(f"Ejecutando query para batch {batch_num}")
        
        # Ejecutar la consulta SQL
        df_batch = duckdb.query(sql).df()

        # Convertir a float32 inmediatamente para ahorrar memoria
        float_cols = df_batch.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            df_batch[float_cols] = df_batch[float_cols].astype('float32')
       
        
        # Mergear con el resultado acumulado
        cols_to_merge = [c for c in df_batch.columns if c not in ['numero_de_cliente', 'foto_mes']]
        
        df_result = df_result.merge(
            df_batch[['numero_de_cliente', 'foto_mes'] + cols_to_merge],
            on=['numero_de_cliente', 'foto_mes'],
            how='left'
        )
        
        logger.info(f"Batch {batch_num} completado. Total columnas: {df_result.shape[1]}")
        
        # Limpiar memoria
        del df_batch
        import gc
        gc.collect()
    
    logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.shape[1]} columnas")
    return df_result




# --------------------------> clase pesada 

def asignar_pesos(df: pd.DataFrame) -> pd.DataFrame:
    """
    'BAJA+2': 2.5,
    'BAJA+1': 1.5,
    'CONTINUA': 1.0
    """

    df['clase_pesada'] = df['clase_ternaria'].map({
        'BAJA+2': 2.5,
        'BAJA+1': 1.5,
        'CONTINUA': 1.0
    })
    return df




# ---------------------> unsersampling para clase 0


def aplicar_undersampling_clase0(
    df: pd.DataFrame,
    undersampling: float,
    id_col: str = 'numero_de_cliente',
    target_col: str = 'clase_ternaria',
    seed: int = 42
) -> pd.DataFrame:
    """
        Aplica undersampling a los clientes que tienen clase 0 en todos los períodos del dataset.
        Mantiene el 100% de los clientes que alguna vez tuvieron '1' y submuestrea el pool de clientes que nunca tuvieron '1' al número
        especificado en 'undersampling'.
        Parameters
    ----------
    df : pd.DataFrame
    undersampling : float
        Proporción de clientes never_1 a conservar (ej: 0.2 = 20%)
    id_col : str, optional
        Nombre de la columna que identifica al cliente.
    target_col : str, optional
        Nombre de la columna target (ej. 'clase_ternaria').
    seed : int, optional
        Semilla para el muestreo reproducible.
    Returns
    -------
    pd.DataFrame
        DataFrame subsampleado.
    """
   
    # Primero calculamos cuántos IDs "never_1" hay
    n_never_1 = duckdb.sql(f"""
        SELECT COUNT(DISTINCT {id_col}) as n
        FROM df
        WHERE {id_col} NOT IN (
            SELECT DISTINCT {id_col}
            FROM df
            WHERE {target_col} = 1
        )
    """).fetchone()[0]
   
    # Determinamos el sample size
    sample_size = int(n_never_1 * undersampling)
   
    # Si sample_size es 0, solo devolver los que tienen clase 1
    if sample_size == 0:
        result = duckdb.sql(f"""
            SELECT df.*
            FROM df
            WHERE {id_col} IN (
                SELECT DISTINCT {id_col}
                FROM df
                WHERE {target_col} = 1
            )
        """).df()
        return result
   
    # Query principal - usando ORDER BY random() con seed
    result = duckdb.sql(f"""
        WITH
        ids_ever_1 AS (
            SELECT DISTINCT {id_col}
            FROM df
            WHERE {target_col} = 1
        ),
        ids_never_1 AS (
            SELECT DISTINCT {id_col}
            FROM df
            WHERE {id_col} NOT IN (SELECT {id_col} FROM ids_ever_1)
        ),
        ids_never_1_sampled AS (
            SELECT {id_col}
            FROM (
                SELECT {id_col}, random() as rnd
                FROM ids_never_1
            )
            ORDER BY rnd
            LIMIT {sample_size}
        ),
        ids_to_keep AS (
            SELECT {id_col} FROM ids_ever_1
            UNION ALL
            SELECT {id_col} FROM ids_never_1_sampled
        )
        SELECT df.*
        FROM df
        INNER JOIN ids_to_keep USING ({id_col})
    """).df()
   
    # Setear la seed antes de ejecutar
    duckdb.execute(f"SELECT setseed({seed / 1000000.0})")
    
    return result



#--------------->

def feature_engineering_lag_delta_polars(
    df: pd.DataFrame, 
    columnas: list[str], 
    cant_lag: int = 1
) -> pd.DataFrame:
    """
    
    """
    logger.info(f"Generando {cant_lag} lags y deltas para {len(columnas)} atributos")
    
    if not columnas:
        return df
    
    if 'numero_de_cliente' not in df.columns or 'foto_mes' not in df.columns:
        raise ValueError("Columnas requeridas no encontradas")
    
    # FILTRAR COLUMNAS QUE NO DEBEN TENER LAGS
    columnas_excluidas = ['clase_ternaria', 
                          'numero_de_cliente', 'foto_mes', 'periodo0']
    
    columnas_validas = [col for col in columnas 
                       if col in df.columns and col not in columnas_excluidas]
    
    if len(columnas_validas) < len(columnas):
        excluidas = set(columnas) - set(columnas_validas)
        logger.warning(f"Columnas excluidas o no encontradas: {excluidas}")
    
    logger.info(f"Procesando {len(columnas_validas)} columnas válidas")
    
    # Convertir a Polars LazyFrame
    logger.info("Convirtiendo a Polars LazyFrame...")
    df_pl = pl.from_pandas(df).lazy()
    
    # Calcular periodo0
    df_pl = df_pl.with_columns(
        ((pl.col("foto_mes") // 100) * 12 + (pl.col("foto_mes") % 100)).alias("periodo0")
    )
    
    # Construir expresiones
    logger.info(f"Construyendo expresiones para {len(columnas_validas)} columnas...")
    expresiones = []
    
    for attr in columnas_validas:
        for j in range(1, cant_lag + 1):
            # Lag
            expresiones.append(
                pl.col(attr)
                .shift(j)
                .over("numero_de_cliente", order_by="periodo0")
                .cast(pl.Float32)
                .alias(f"{attr}_lag_{j}")
            )
            # Delta
            expresiones.append(
                (pl.col(attr) - pl.col(attr).shift(j).over("numero_de_cliente", order_by="periodo0"))
                .cast(pl.Float32)
                .alias(f"{attr}_delta_{j}")
            )
    
    # Aplicar transformaciones
    logger.info(f"Aplicando {len(expresiones)} transformaciones...")
    df_pl = df_pl.with_columns(expresiones)
    
    # Ejecutar
    logger.info("Ejecutando query optimizada...")
    df_result = df_pl.collect().to_pandas()
    
    import gc
    gc.collect()
    
    logger.info(f"Completado: {df_result.shape}")
    logger.info(f"Nuevas columnas creadas: {len(expresiones)}")
    
    return df_result


# ---------------------->  Medias móviles

def feature_engineering_rolling_mean(df, columnas_validas, ventana=3):
    logger.info(f"Realizando medias móviles a {len(columnas_validas)} columnas. Promedio de {len(ventana)} meses ")
    
    df_pl = pl.from_pandas(df).lazy()

    # Crear periodo0 si no existe
    if 'periodo0' not in df.columns:
        df_pl = df_pl.with_columns(
            ((pl.col("foto_mes") // 100) * 12 + (pl.col("foto_mes") % 100)).alias("periodo0"))
        drop_periodo0 = True
    else:
        drop_periodo0 = False

    # Expresiones rolling
    expresiones = [
        pl.col(col).shift(1)
        .rolling_mean(window_size=ventana, min_periods=1)
        .over("numero_de_cliente", order_by="periodo0")
        .alias(f"{col}_rolling_mean_{ventana}")
        for col in columnas_validas
    ]

    df_pl = df_pl.with_columns(expresiones)

    if drop_periodo0:
        df_pl = df_pl.drop("periodo0")

    df_result = df_pl.collect().to_pandas()
    gc.collect()
    return df_result