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
        df = pl.read_csv(path, infer_schema_length=None)
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

