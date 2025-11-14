import logging
import pandas as pd
import polars as pl
import os
import datetime
import sys
print(sys.executable)

import os
import duckdb
import numpy as np 

logger = logging.getLogger(__name__)


import polars as pl
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def cargar_datos(path: str) -> pl.DataFrame | None:
    '''
    Carga un CSV desde 'path' y retorna un Polars DataFrame.
    Si falla la lectura directa con Polars, reintenta con Pandas y convierte a Polars.
    '''
    logger.info(f"Cargando dataset desde {path}")

    try:
        df = pl.read_csv(path, infer_schema_length=10000)
        logger.info(f"Dataset cargado con Polars: {df.height} filas y {df.width} columnas")
        return df

    except Exception as e:
        logger.warning(f"Fallo lectura directa con Polars: {e}")
        logger.info("Reintentando lectura con Pandas...")

        try:
            # Segundo intento: Pandas (más tolerante con tipos mixtos)
            df_pd = pd.read_csv(path, low_memory=False)
            logger.info(f"Dataset cargado con Pandas: {df_pd.shape[0]} filas y {df_pd.shape[1]} columnas")

            # Conversión a Polars
            df_pl = pl.from_pandas(df_pd)
            logger.info(f"Convertido a Polars: {df_pl.height} filas y {df_pl.width} columnas")

            return df_pl

        except Exception as e2:
            logger.error(f"Error al cargar el dataset con Pandas: {e2}")
            raise



#----------------------> creación de clase ternaria a partir de la identificación de los períodos

def crear_clase_ternaria(df: pl.DataFrame) -> pd.DataFrame:
    """
    Crea la clase ternaria para un DataFrame de clientes usando DuckDB.
    Devuelve el DataFrame original con columna 'clase_ternaria'.
    """
    logger.info("Definiendo variable target")

    # Conexión en memoria
    conn = duckdb.connect(database=':memory:', read_only=False)

    # Registro del DataFrame como tabla temporal
    conn.register("df_tmp", df)

    # Crear la clase ternaria
    conn.execute("""
    CREATE OR REPLACE TABLE clases AS
    WITH periodos AS (
        SELECT 
            numero_de_cliente,
            foto_mes,
            CAST(foto_mes / 100 AS INTEGER) * 12 + (foto_mes % 100) AS periodo0
        FROM df_tmp
    ),
    max_periodo AS (
        SELECT MAX(periodo0) AS max_periodo0 FROM periodos
    )
    SELECT
        d.numero_de_cliente,
        d.foto_mes,
        CASE
            WHEN d.periodo0 <= m.max_periodo0 - 1
                AND NOT EXISTS (
                    SELECT 1
                    FROM periodos p
                    WHERE p.numero_de_cliente = d.numero_de_cliente
                    AND p.periodo0 = d.periodo0 + 1
                ) THEN 'BAJA+1'
            WHEN d.periodo0 <= m.max_periodo0 - 2
                AND NOT EXISTS (
                    SELECT 1
                    FROM periodos p
                    WHERE p.numero_de_cliente = d.numero_de_cliente
                    AND p.periodo0 = d.periodo0 + 2
                ) THEN 'BAJA+2'
            ELSE 'CONTINUA'
        END AS clase_ternaria
    FROM periodos d
    CROSS JOIN max_periodo m
    """)

    # Resumen por clase y mes
    resumen_clase_mes = conn.execute("""
        SELECT 
            clase_ternaria,
            foto_mes,
            COUNT(*) AS cantidad_clientes
        FROM clases
        GROUP BY clase_ternaria, foto_mes
        ORDER BY clase_ternaria, foto_mes
    """).pl()

    pivot_df = resumen_clase_mes.pivot(index='clase_ternaria', columns='foto_mes', values='cantidad_clientes').fill_null(0)
    print(pivot_df)

    # Join con el DataFrame original
    df_final = conn.execute("""
        SELECT 
            df_tmp.*,
            c.clase_ternaria
        FROM df_tmp
        LEFT JOIN clases c
            ON df_tmp.numero_de_cliente = c.numero_de_cliente
           AND df_tmp.foto_mes = c.foto_mes
    """).pl()

    # Log resumen final
    n_continua = (df_final.filter(pl.col("clase_ternaria") == "CONTINUA")).height
    n_baja1 = (df_final.filter(pl.col("clase_ternaria") == "BAJA+1")).height
    n_baja2 = (df_final.filter(pl.col("clase_ternaria") == "BAJA+2")).height
    total_bajas = n_baja1 + n_baja2

    logger.info(f"Clase ternaria creada: CONTINUA={n_continua}, BAJA+1={n_baja1}, BAJA+2={n_baja2}, Total bajas={total_bajas}")

    return df_final


#-------------------> Convertir clase ternaria a target con identificación de los tipos bajas

def convertir_clase_ternaria_a_target(df: pd.DataFrame, baja_2_1=True) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario usando SQL en DuckDB:
    - CONTINUA = 0
    - Si baja_2_1=True: BAJA+1 y BAJA+2 = 1
    - Si baja_2_1=False: BAJA+1 = 0 y BAJA+2 = 1
    
    Args:
        df: DataFrame con columna 'clase_ternaria'
        baja_2_1: Booleano que indica si se considera BAJA+1 como positivo
        
    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """
    # Conexión en memoria
    conn = duckdb.connect(database=':memory:', read_only=False)
   
    # Crear tabla temporal a partir del DataFrame
    conn.register("df_tmp", df)
    
    # Contar valores originales para logging
    valores_orig = conn.execute("""
        SELECT 
            clase_ternaria,
            COUNT(*) as count
        FROM df_tmp
        GROUP BY clase_ternaria
    """).df()
    
    n_continua_orig = valores_orig[valores_orig['clase_ternaria'] == 'CONTINUA']['count'].sum()
    n_baja1_orig = valores_orig[valores_orig['clase_ternaria'] == 'BAJA+1']['count'].sum()
    n_baja2_orig = valores_orig[valores_orig['clase_ternaria'] == 'BAJA+2']['count'].sum()
   
    # Convertir con SQL usando CASE según baja_2_1
    if baja_2_1:
        query = """
            SELECT * EXCLUDE (clase_ternaria),
                CASE
                    WHEN clase_ternaria = 'CONTINUA' THEN 0
                    ELSE 1
                END AS clase_ternaria
            FROM df_tmp
        """
    else:
        query = """
            SELECT * EXCLUDE (clase_ternaria),
                CASE
                    WHEN clase_ternaria = 'BAJA+2' THEN 1
                    ELSE 0
                END AS clase_ternaria
            FROM df_tmp
        """
    
    df_result = conn.execute(query).df()
   
    # Log de la conversión
    n_ceros = (df_result['clase_ternaria'] == 0).sum()
    n_unos = (df_result['clase_ternaria'] == 1).sum()
    total = n_ceros + n_unos
   
    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos/total*100:.2f}% casos positivos")
   
    conn.close()
    
    return df_result



# ---------------- > clase ternaria con polars

def convertir_clase_ternaria_a_target_polars(df: pl.DataFrame, baja_2_1: bool = True) -> pl.DataFrame:
    """Conversión binaria optimizada de clase_ternaria (segura y eficiente)."""
    
    # Conteos originales
    conteos = df.group_by("clase_ternaria").agg(pl.count().alias("count"))
    conteos_dict = {row["clase_ternaria"]: row["count"] for row in conteos.iter_rows(named=True)}
    
    n_continua = conteos_dict.get("CONTINUA", 0)
    n_baja1 = conteos_dict.get("BAJA+1", 0)
    n_baja2 = conteos_dict.get("BAJA+2", 0)
    
    # Conversión directa
    if baja_2_1:
        df = df.with_columns(
            (pl.col("clase_ternaria") != "CONTINUA").cast(pl.Int8).alias("clase_ternaria")
        )
    else:
        df = df.with_columns(
            (pl.col("clase_ternaria") == "BAJA+2").cast(pl.Int8).alias("clase_ternaria")
        )
    
    # Conteos finales
    n_unos = int(df["clase_ternaria"].sum())
    n_ceros = len(df) - n_unos
    
    logger.info("Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua}, BAJA+1: {n_baja1}, BAJA+2: {n_baja2}")
    logger.info(f"  Binario  - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos / len(df) * 100:.2f}% casos positivos")
    
    return df

#---------------------------> carga de datos y undersampling si aplica

def preparar_datos_training_lgb(
        df: pl.DataFrame,
        training: list | int,
        validation: list | int,
        undersampling_0: int = 1
    ):
    """
    Prepara datos de entrenamiento y validación para LightGBM
    de forma consistente con la función de entrenamiento final.

    Args:
        df: Polars DataFrame completo
        training: lista o entero con períodos de training
        validation: lista o entero con períodos de validación
        undersampling_0: cada cuántos registros clase 0 dejar (1 = no undersampling)

    Returns:
        tuple: (lgb_train, lgb_val, X_train, y_train, X_val, y_val)
    """

    logger.info("Preparando datos para entrenamiento + validación")

    if isinstance(training, list): 
        df_train = df.filter(pl.col('foto_mes').is_in(training))
    else:
        df_train = df.filter(pl.col('foto_mes') == training)

    if isinstance(validation, list):
        df_val = df.filter(pl.col('foto_mes').is_in(validation))
    else:
        df_val = df.filter(pl.col('foto_mes') == validation)

    
    logger.info(f"Tamaño original train: {len(df_train):,} | "f"Períodos train: {training}")
    logger.info(f"Tamaño val: {len(df_val):,} | "f"Períodos val: {validation}")


    if df_train.is_empty():
        raise ValueError(f"No se encontraron datos de training para períodos: {training}")
    if df_val.is_empty():
        raise ValueError(f"No se encontraron datos de validation para períodos: {validation}")

    df_train = convertir_clase_ternaria_a_target_polars(df_train, baja_2_1=True)
    df_val = convertir_clase_ternaria_a_target_polars(df_val, baja_2_1=False)

    df_train = aplicar_undersampling_clase0(df_train, undersampling_0, seed=SEMILLAS[0])

    logger.info(f"Train luego de undersampling: {len(df_train):,}")

    X_train = df_train.drop('clase_ternaria')
    y_train = df_train['clase_ternaria'].to_numpy()

    X_val = df_val.drop('clase_ternaria')
    y_val = df_val['clase_ternaria'].to_numpy()

    vc_train = df_train['clase_ternaria'].value_counts().to_dict()
    logger.info("Distribución training:")
    for clase, count in vc_train.items():
        logger.info(f"  Clase {clase}: {count:,} ({count/len(df_train)*100:.0f}%)")

    vc_val = df_val['clase_ternaria'].value_counts().to_dict()
    logger.info("Distribución validation:")
    for clase, count in vc_val.items():
        logger.info(f"  Clase {clase}: {count:,} ({count/len(df_val)*100:.0f}%)")

    lgb_train = lgb.Dataset(X_train.to_pandas(), label=y_train)
    lgb_val = lgb.Dataset(X_val.to_pandas(), label=y_val, reference=lgb_train)

    return lgb_train, lgb_val, X_train, y_train, X_val, y_val



def preparar_datos_training_lgb(df, training: list, validation:list, undersampling_0: int = 1) -> pl.Dataframe: 
    """

    """

    if isinstance(training, list):
        df_train = df.filter(pl.col('foto_mes').is_in(training))
    else:
        df_train = df.filter(pl.col('foto_mes') == training)
    
    df_val = df.filter(pl.col('foto_mes') == validation)

    
    logger.info(
        f"Tamaño original de train: {len(df_train)}. "
        f"Rango train: {min(training) if isinstance(training, list) else training} - "
        f"{max(training) if isinstance(training, list) else training}. "
        f"Tamaño val: {len(df_val)}. Val: {validation}")


    #Convierto a binaria la clase ternaria 
    df_train = convertir_clase_ternaria_a_target_polars(df_train, baja_2_1=True) # Entreno el modelo con Baja+1 y Baja+2 == 1
    df_val = convertir_clase_ternaria_a_target_polars(df_val, baja_2_1=False) # valido la ganancia solamente con Baja+2 == 1
    

    #Subsampleo
    df_train = aplicar_undersampling_clase0(df_train, undersampling, seed= SEMILLAS[0])   #VER SEMILLA.
    n_train = len(df_train)

    # Separar features y target
    X_train = df_train.drop('clase_ternaria')
    y_train = df_train['clase_ternaria'].to_numpy()
    lgb_train = lgb.Dataset(X_train, label=y_train)
    
    X_val = df_val.drop('clase_ternaria')
    y_val = df_val['clase_ternaria'].to_numpy()
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
