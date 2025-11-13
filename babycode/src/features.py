# src/features.py
import pandas as pd
import duckdb
import logging
import re
import sys
print(sys.executable)
import numpy as np

import os
import duckdb
import pandas as pd 
import polars as pl
from .loader import crear_clase_ternaria 
import gc



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

def select_col_montos(df: pl.DataFrame) -> list[str]:
    """"
    Selecciona columnas de "montos" de un DataFrame según patrones:
      - columnas que empiezan con 'm'
      - columnas que contienen 'Master_m' o 'Visa_m'
    Siempre excluye: 'numero_de_cliente' y 'foto_mes'
    
    Parámetros:
        df: DataFrame de Polars
    Retorna:
        list: columnas seleccionadas
    """
    import polars.selectors as cs
    
    # Seleccionar columnas que empiezan con 'm' o contienen los patrones
    selected = df.select(
        cs.starts_with("m") | cs.contains("Master_m") | cs.contains("Visa_m")
    ).columns
    
    # Excluir columnas específicas
    excluded = {"numero_de_cliente", "foto_mes"}
    return [col for col in selected if col not in excluded]

#-----------------------------------------------------> Rank positivo batch

def feature_engineering_rank_pos_batch(df: pl.DataFrame, columnas: list[str]) -> pl.DataFrame:
    """
    Genera rankings normalizados por signo para las columnas especificadas,
    procesando los datos por 'foto_mes' y columna para reducir el uso de memoria.
    """
    import duckdb
    import polars as pl
    
    if not columnas:
        raise ValueError("La lista de columnas no puede estar vacía")
    
    columnas_validas = [col for col in columnas if col in df.columns]
    if not columnas_validas:
        raise ValueError("No hay columnas válidas para rankear")
    
    logger.info(f"Generando ranking por batch: {len(columnas_validas)} columnas válidas.")
    
    # Convertir Polars a Pandas para DuckDB
    df_pandas = df.to_pandas()
    
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads=1;")
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET memory_limit='2GB';")
    
    df_result = []
    
    for mes in sorted(df_pandas["foto_mes"].unique()):
        df_mes = df_pandas[df_pandas["foto_mes"] == mes].copy()
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
    
    resultado_pandas = pd.concat(df_result, ignore_index=True)
    
    # Convertir de vuelta a Polars
    resultado = pl.from_pandas(resultado_pandas)
    
    logger.info(f"Feature engineering completado por batch y columna. Shape final: {resultado.shape}")
    
    return resultado


#-------------------------------> Crea LAG y DELTA en batch.

def feature_engineering_lag_delta_batch(
    df: pl.DataFrame, 
    columnas: list[str], 
    cant_lag: int = 1, 
    batch_size: int = 25
) -> pl.DataFrame:
    """
    Genera variables de lag y delta para los atributos especificados utilizando DuckDB.
    Procesa columnas en batches para evitar problemas de memoria.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame con los datos originales.
    columnas : list[str]
        Lista de atributos para los cuales generar lags y deltas.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo.
    batch_size : int, default=25
        Número de columnas a procesar por batch.
    
    Returns
    -------
    pl.DataFrame
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
    
    # Configurar DuckDB
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads TO 4;")
    con.execute("SET memory_limit='4GB';")
    
    # Iniciar con el DataFrame original
    df_result = df.clone()
    
    # Procesar columnas en batches
    num_batches = (len(columnas) - 1) // batch_size + 1
    
    for i in range(0, len(columnas), batch_size):
        batch_num = i // batch_size + 1
        batch_cols = columnas[i:i+batch_size]
        
        logger.info(f"Procesando batch {batch_num}/{num_batches}: {len(batch_cols)} columnas")
        
        # Registrar DataFrame en DuckDB
        con.register("df_temp", df)
        
        # Construir la consulta SQL solo para este batch
        sql_parts = ["numero_de_cliente", "foto_mes"]
        
        for attr in batch_cols:
            if attr in df.columns:
                for j in range(1, cant_lag + 1):
                    # Escapar nombres de columnas con comillas dobles
                    sql_parts.append(f'LAG("{attr}", {j}) OVER w AS "{attr}_lag_{j}"')
                    sql_parts.append(f'("{attr}" - LAG("{attr}", {j}) OVER w) AS "{attr}_delta_{j}"')
            else:
                logger.warning(f"El atributo {attr} no existe en el DataFrame")
        
        sql = f"""
        SELECT {', '.join(sql_parts)}
        FROM df_temp 
        WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)
        """
        
        logger.debug(f"Ejecutando query para batch {batch_num}")
        
        # Ejecutar la consulta SQL y convertir a Polars
        df_batch = con.execute(sql).pl()
        
        # Convertir float64 a float32 para ahorrar memoria
        float_cols = [col for col in df_batch.columns 
                     if df_batch[col].dtype == pl.Float64]
        if float_cols:
            df_batch = df_batch.with_columns([
                pl.col(c).cast(pl.Float32) for c in float_cols
            ])
        
        # Unregister para liberar memoria
        con.unregister("df_temp")
        
        # Mergear con el resultado acumulado
        cols_to_merge = [c for c in df_batch.columns 
                        if c not in ['numero_de_cliente', 'foto_mes']]
        
        df_result = df_result.join(
            df_batch.select(['numero_de_cliente', 'foto_mes'] + cols_to_merge),
            on=['numero_de_cliente', 'foto_mes'],
            how='left'
        )
        
        logger.info(f"Batch {batch_num} completado. Total columnas: {df_result.width}")
        
        # Limpiar memoria
        del df_batch
        gc.collect()
    
    con.close()
    
    logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.width} columnas")
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
    df: pl.DataFrame,
    undersampling: float,
    id_col: str = 'numero_de_cliente',
    target_col: str = 'clase_ternaria',
    seed: int = 42
) -> pl.DataFrame:
    """
    Aplica undersampling a los clientes que tienen clase 0 en todos los períodos del dataset.
    Mantiene el 100% de los clientes que alguna vez tuvieron '1' y submuestrea el pool de clientes
    que nunca tuvieron '1' al número especificado en 'undersampling'.

    Parameters
    ----------
    df : pl.DataFrame
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
    pl.DataFrame
        DataFrame subsampleado.
    """

    # IDs que alguna vez tuvieron clase 1
    ids_ever_1 = (
        df.filter(pl.col(target_col) == 1)
          .select(id_col)
          .unique()
    )

    # IDs que nunca tuvieron clase 1
    ids_never_1 = (
        df.select(id_col)
          .unique()
          .join(ids_ever_1, on=id_col, how="anti")
    )

    n_never_1 = ids_never_1.height
    sample_size = int(n_never_1 * undersampling)

    # Si sample_size es 0, devolver solo los que tienen clase 1
    if sample_size == 0:
        result = df.join(ids_ever_1, on=id_col, how="inner")
        logger.info(f"Undersampling clase 0: {undersampling:.2%} (semilla={seed}) — solo clientes con clase 1")
        return result

    np.random.seed(seed)
    sampled_ids = np.random.choice(
        ids_never_1[id_col].to_list(),
        size=sample_size,
        replace=False)

    ids_never_1_sampled = pl.DataFrame({id_col: sampled_ids})

    # Combinar ids a conservar
    ids_to_keep = pl.concat([ids_ever_1, ids_never_1_sampled])

    # Filtrar el dataset original
    result = df.join(ids_to_keep, on=id_col, how="inner")

    logger.info(f"Undersampling clase 0: {undersampling:.2%} (semilla={seed}) "
                f"— {sample_size} de {n_never_1} clientes 'never_1' conservados")

    return result


#--------------->

def feature_engineering_lag_delta_polars(
    df: pl.DataFrame, 
    columnas: list[str], 
    cant_lag: int = 1
) -> pl.DataFrame:
    """
    Genera variables de lag y delta usando Polars puro.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame con los datos originales.
    columnas : list[str]
        Lista de atributos para los cuales generar lags y deltas.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo.
    
    Returns
    -------
    pl.DataFrame
        DataFrame con las variables de lag y delta agregadas.
    """
    logger.info(f"Generando {cant_lag} lags y deltas para {len(columnas)} atributos")
    
    if not columnas:
        return df
    
    if 'numero_de_cliente' not in df.columns or 'foto_mes' not in df.columns:
        raise ValueError("Columnas requeridas no encontradas")
    
    columnas_excluidas = ['clase_ternaria', 'numero_de_cliente', 'foto_mes', 'periodo0']
    
    # Filtrar solo columnas numéricas
    tipos_numericos = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                       pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                       pl.Float32, pl.Float64]
    
    columnas_validas = [
        col for col in columnas 
        if col in df.columns 
        and col not in columnas_excluidas
        and df.schema[col] in tipos_numericos
    ]
    
    if len(columnas_validas) < len(columnas):
        excluidas = set(columnas) - set(columnas_validas)
        logger.warning(f"Columnas excluidas o no encontradas: {excluidas}")
    
    logger.info(f"Procesando {len(columnas_validas)} columnas válidas")
    
    # Trabajar con LazyFrame para optimización
    df_pl = df.lazy()
    
    # Calcular periodo0 si no existe
    if 'periodo0' not in df.columns:
        df_pl = df_pl.with_columns(
            ((pl.col("foto_mes") // 100) * 12 + (pl.col("foto_mes") % 100)).alias("periodo0")
        )
    
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
    df_pl = df_pl.with_columns(expresiones)
    logger.info("Ejecutando query")
    df_result = df_pl.collect()
    
    gc.collect()
    
    logger.info(f"Completado: {df_result.shape}")
    logger.info(f"Nuevas columnas creadas: {len(expresiones)}")
    
    return df_result

# ---------------------->  Medias móviles

def feature_engineering_rolling_mean(df: pl.DataFrame, columnas_validas: list[str], ventana: int = 3) -> pl.DataFrame:
    """
    Calcula medias móviles para las columnas especificadas usando Polars puro.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame con los datos originales.
    columnas_validas : list[str]
        Lista de columnas para calcular rolling mean.
    ventana : int, default=3
        Tamaño de la ventana (en meses).
    
    Returns
    -------
    pl.DataFrame
        DataFrame con las columnas de rolling mean agregadas.
    """
    import gc
    
    logger.info(f"Realizando medias móviles a {len(columnas_validas)} columnas. Promedio de {ventana} meses")
    
    df_pl = df.lazy()
    
    # Crear periodo0 si no existe
    if 'periodo0' not in df.columns:
        df_pl = df_pl.with_columns(
            ((pl.col("foto_mes") // 100) * 12 + (pl.col("foto_mes") % 100)).alias("periodo0")
        )
        drop_periodo0 = True
    else:
        drop_periodo0 = False
    
    # Expresiones rolling
    expresiones = [
        pl.col(col)
        .shift(1)
        .rolling_mean(window_size=ventana, min_periods=1)
        .over("numero_de_cliente", order_by="periodo0")
        .alias(f"{col}_rolling_mean_{ventana}")
        for col in columnas_validas
    ]
    
    df_pl = df_pl.with_columns(expresiones)
    
    # Eliminar periodo0 si fue creado aquí
    if drop_periodo0:
        df_pl = df_pl.drop("periodo0")

    df_result = df_pl.collect()

    logger.info(f"Completado: {df_result.height:,} filas x {df_result.width:,} columnas")
    
    gc.collect()
    
    return df_result



# -------------------------------> reemplazo de ceros
def zero_replace(df: pl.DataFrame, group_col: str = "foto_mes") -> pl.DataFrame:
    """Reemplaza con NAN aquellas columnas que tienen 0 en todos sus valores"""
    
    # Filtrar solo columnas numéricas y excluir group_col
    cols = [
        c for c in df.columns 
        if c != group_col and df[c].dtype.is_numeric()
    ]
    
    exprs = [
        pl.when((pl.col(c) == 0).all().over(group_col) & (pl.col(c) == 0))
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in cols
    ]
    
    return df.with_columns(exprs)


# -------------------------------> Eliminar meses

def filtrar_meses(df, col='foto_mes', mes_inicio=202003, mes_fin=202007):
    """
    Devuelve un DataFrame con solo los meses dentro del rango [mes_inicio, mes_fin].
    """
    # Mask booleana: True si el mes está dentro del rango
    mask = (df[col] >= mes_inicio) & (df[col] <= mes_fin)
    
    # Filtramos y devolvemos
    return df.loc[mask].copy()


# -------------------> Neutralizacion de columnas

def neutral_columns(df: pl.DataFrame, columnas: list[str]) -> pl.DataFrame:
    """
    Rellena las columnas indicadas con null (NaN en Polars).
    
    Parámetros:
    - df: DataFrame original
    - columnas: lista de nombres de columnas a rellenar
    
    Devuelve:
    - DataFrame modificado (las columnas especificadas ahora contienen solo null)
    """
    # Filtrar solo columnas que existen en el DataFrame
    cols_existentes = [c for c in columnas if c in df.columns]
    
    # Crear expresiones que reemplazan con null
    return df.with_columns([pl.lit(None).alias(c) for c in cols_existentes])

# -------------> drop columns

def drop_columns(df: pl.DataFrame, columnas: list[str]) -> pl.DataFrame:
    """
    Elimina las columnas indicadas del DataFrame.
    
    Parámetros:
    - df: DataFrame original
    - columnas: lista de nombres de columnas a eliminar
    
    Devuelve:
    - DataFrame sin las columnas especificadas
    """
    # Filtrar solo columnas que existen en el DataFrame
    cols_existentes = [c for c in columnas if c in df.columns]
    
    return df.drop(cols_existentes)


# -------------> reg_slope

def feature_engineering_reg(df, columnas: list[str]) -> pl.DataFrame:
    """
    Genera reg_slope para los atributos especificados utilizando SQL.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar tendencias. Si es None, no se generan tendencias.

    Returns:
    --------
    pl.DataFrame
        DataFrame con las variables de tendencia agregadas
    """

    logger.info(f"Realizando feature engineering de tendencia para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    # Construir la consulta SQL
    sql = "SELECT *"

    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            sql += f", regr_slope({attr}, cliente_antiguedad) over ventana as {attr}_trend_6m"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")

    # Completar la consulta
    sql += " FROM df"
    sql += " window ventana as (partition by numero_de_cliente order by foto_mes rows between 6 preceding and current row)"
    sql += " ORDER BY numero_de_cliente, foto_mes"

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).pl()
    con.close()


    df_result = df_pl.collect().to_pandas()
    
    logging.info(df.head())

    logger.info(f"Feature engineering [trends] completado")
    logger.info(df.shape)

    return df