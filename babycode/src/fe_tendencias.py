import polars as pl
import numpy as np
from numba import njit

def calcular_vector_desde(df, ventana):
    n = df.height
    ids = df["numero_de_cliente"].to_numpy()
    
    desde = np.zeros(n, dtype=np.int64)
    for i in range(n):
        desde[i] = i - ventana + 2
        
    for i in range(min(n, ventana)):
        desde[i] = 1

    for i in range(1, n):
        if ids[i-1] != ids[i]:
            desde[i] = i + 1

    for i in range(1, n):
        if desde[i] < desde[i-1]:
            desde[i] = desde[i-1]

    return desde



@njit
def fhistC_py_numba(columna, desde):
    n = len(columna)

    out = np.full(5*n, np.nan, dtype=np.float64)
    x = np.zeros(100, dtype=np.float64)
    y = np.zeros(100, dtype=np.float64)


    for i in range(n):

        if desde[i] - 1 < i:
            out[i + 4*n] = columna[i-1]
        else:
            out[i + 4*n] = np.nan

        libre = 0
        xvalor = 1
        for j in range(desde[i] - 1, i + 1):
            a = columna[j]

            if not np.isnan(a):
                y[libre] = a
                x[libre] = xvalor
                libre += 1
            xvalor += 1

        if libre > 1:
            xsum = x[:libre].sum()
            ysum = y[:libre].sum()
            xysum = (x[:libre] * y[:libre]).sum()
            xxsum = (x[:libre] * x[:libre]).sum()
            vmin = y[:libre].min()
            vmax = y[:libre].max()

            # Cálculo de la pendiente (fórmula inalterada)
            pendiente = (libre * xysum - xsum * ysum) / (libre * xxsum - xsum * xsum)

            out[i] = pendiente
            out[i+n] = vmin
            out[i+2*n] = vmax
            out[i+3*n] = ysum / libre
        else:
            # Asignación de NaN
            out[i] = np.nan
            out[i+n] = np.nan
            out[i+2*n] = np.nan
            out[i+3*n] = np.nan

    return out




def tendencia_polars(
    df,
    cols,
    ventana=6,
    tendencia=True,
    minimo=True,
    maximo=True,
    promedio=True,
    ratioavg=False,
    ratiomax=False
):
    df = df.sort(["numero_de_cliente", "foto_mes"])

    # 1. calcular vector_desde igual que en R
    vector_desde = calcular_vector_desde(df, ventana)
    n = df.height
    
    # 2. por cada columna, aplicar fhistC_py y agregar columnas
    for col in cols:

        # ejecutar algoritmo C++ traducido
        out = fhistC_py_numba(df[col].to_numpy(), vector_desde)

        # separar los 5 bloques
        ind = np.arange(n)
        
        if tendencia:
            df = df.with_columns(pl.Series(f"{col}_tend{ventana}", out[0*n : 1*n]))

        if minimo:
            df = df.with_columns(pl.Series(f"{col}_min{ventana}", out[1*n : 2*n]))

        if maximo:
            df = df.with_columns(pl.Series(f"{col}_max{ventana}", out[2*n : 3*n]))

        if promedio:
            df = df.with_columns(pl.Series(f"{col}_avg{ventana}", out[3*n : 4*n]))

        if ratioavg:
            df = df.with_columns(
                pl.Series(f"{col}_ratioavg{ventana}", df[col].to_numpy() / out[3*n : 4*n])
            )

        if ratiomax:
            df = df.with_columns(
                pl.Series(f"{col}_ratiomax{ventana}", df[col].to_numpy() / out[2*n : 3*n])
            )
            
    return df

