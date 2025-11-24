import polars as pl
import numpy as np

def calcular_vector_desde(df, ventana):
    n = df.height()
    ids = df["numero_de_cliente"].to_numpy()
    desde = np.arange(-ventana + 2, n - ventana + 2)
    desde[:ventana] = 1

    for i in range(1, n):
        if ids[i-1] != ids[i]:
            desde[i] = i+1

    for i in range(1, n):
        if desde[i] < desde[i-1]:
            desde[i] = desde[i-1]

    return desde


def fhistC_py(columna, desde):
    columna = np.array(columna, dtype=float)
    desde = np.array(desde, dtype=int)
    n = len(columna)

    out = np.full(5*n, np.nan, dtype=float)
    x = np.zeros(100, dtype=float)
    y = np.zeros(100, dtype=float)

    for i in range(n):
        # lag
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
            xs = x[:libre]
            ys = y[:libre]
            xsum = xs.sum()
            ysum = ys.sum()
            xysum = (xs * ys).sum()
            xxsum = (xs * xs).sum()
            vmin = ys.min()
            vmax = ys.max()

            pendiente = (libre * xysum - xsum * ysum) / (libre * xxsum - xsum * xsum)

            out[i] = pendiente
            out[i+n] = vmin
            out[i+2*n] = vmax
            out[i+3*n] = ysum / libre
        else:
            out[i:i+4*n:n] = np.nan

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
    df = df.with_columns(pl.Series("vector_desde", vector_desde))

    n = df.height()
    
    # 2. por cada columna, aplicar fhistC_py y agregar columnas
    for col in cols:

        # ejecutar algoritmo C++ traducido
        out = fhistC_py(df[col].to_numpy(), df["vector_desde"].to_numpy())

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

    # quitar vector_desde 
    df = df.drop("vector_desde")
    return df

