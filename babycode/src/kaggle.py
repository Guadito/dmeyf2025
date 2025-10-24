import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from src.gain_function import calcular_ganancia

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

def simular_cortes_kaggle(y_pred_prob: np.ndarray,
                          y_test: pd.Series,
                          cortes: list,
                          ganancia_por_corte,
                          n_splits: int = 50,
                          test_size: float = 0.3,
                          random_state: int = 42):

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    y_test_array = y_test.to_numpy()

    rows = []
    print(f"Iniciando {n_splits} simulaciones para {len(cortes)} cortes...")

    for i, (private_idx, public_idx) in enumerate(sss.split(np.zeros(len(y_test_array)), y_test_array)):
        if (i + 1) % 10 == 0:
            print(f"  Simulación {i+1}/{n_splits}...")

        row = {}
        for corte in cortes:
            g_public = ganancia_por_corte(y_pred_prob[public_idx], y_test_array[public_idx], corte)
            g_private = ganancia_por_corte(y_pred_prob[private_idx], y_test_array[private_idx], corte)

            row[f"corte_{corte}_public"] = g_public
            row[f"corte_{corte}_private"] = g_private

        rows.append(row)

    print("Simulación completada.")
    return pd.DataFrame(rows)




def ganancia_por_corte(y_prob, y_true_full, n_envios):
    """
    Calcula la ganancia para un número específico de envíos (corte).
    """
    n_envios = min(n_envios, len(y_prob))
    indices_ordenados = np.argsort(y_prob)[::-1]
    indices_top_n = indices_ordenados[:n_envios]
    y_pred = np.zeros_like(y_prob, dtype=int)
    y_pred[indices_top_n] = 1

    ganancia = calcular_ganancia(y_true_full, y_pred)
    return ganancia




def resumen_cortes(df_lb):
    """
    Devuelve una tabla resumen con la ganancia promedio por corte (public/private)
    y la diferencia porcentual entre ambas. Sin prints ni gráficos.
    """
    df_resumen = (
        df_lb.mean()
        .to_frame("ganancia_media")
        .reset_index()
    )

    df_resumen["tipo"] = df_resumen["index"].apply(lambda x: "public" if "public" in x else "private")
    df_resumen["corte"] = df_resumen["index"].str.extract(r"(\d+)")
    
    df_resumen = (
        df_resumen
        .pivot(index="corte", columns="tipo", values="ganancia_media")
        .reset_index()
        .sort_values("corte")
    )

    df_resumen["delta_%"] = 100 * (df_resumen["private"] - df_resumen["public"]) / df_resumen["public"]

    return df_resumen

