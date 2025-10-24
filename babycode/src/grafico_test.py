# src/grafico_test.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
from .config import STUDY_NAME, GANANCIA_ACIERTO, COSTO_ESTIMULO, MES_TEST
from .gain_function import *

logger = logging.getLogger(__name__)




#------------------------> Gráfico de ganancia avanzado

def crear_grafico_ganancia_avanzado(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   titulo_personalizado: str = None) -> str:
    """
    Crea un gráfico avanzado de ganancia acumulada con múltiples elementos informativos.
  
    Args:
        y_true: Valores verdaderos
        y_pred_proba: Probabilidades predichas
        titulo_personalizado: Título personalizado para el gráfico
  
    Returns:
        str: Ruta del archivo del gráfico guardado
    """
    logger.info("Generando gráfico de ganancia avanzado...")
  
    # Calcular ganancia acumulada
    ganancias_acumuladas, indices_ordenados, umbral_optimo = calcular_ganancia_acumulada_optimizada(y_true, y_pred_proba)
  
    # Encontrar estadísticas clave
    ganancia_maxima = np.max(ganancias_acumuladas)
    indice_maximo = np.argmax(ganancias_acumuladas)
  
    # Calcular puntos de referencia
    umbral_025 = 0.025
    clientes_sobre_025 = np.sum(y_pred_proba >= umbral_025)
  
    # Filtrar datos para visualización (solo mostrar región relevante)
    umbral_ganancia = ganancia_maxima * 0.6  # Mostrar desde 60% de la ganancia máxima
    indices_relevantes = ganancias_acumuladas >= umbral_ganancia
    x_relevante = np.where(indices_relevantes)[0]
    y_relevante = ganancias_acumuladas[indices_relevantes]
  
    # Configurar estilo del gráfico
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
  
    # Gráfico principal: Ganancia acumulada
    ax1.plot(x_relevante, y_relevante, color='blue', linewidth=3, label='Ganancia Acumulada', alpha=0.8)
  
    # Marcar ganancia máxima
    ax1.scatter(indice_maximo, ganancia_maxima, color='red', s=150, zorder=5, 
               label=f'Ganancia Máxima: {ganancia_maxima:,.0f}')
  
    # Líneas de referencia
    ax1.axvline(x=indice_maximo, color='red', linestyle='--', alpha=0.7, 
               label=f'Corte Óptimo (cliente {indice_maximo:,})')
    ax1.axvline(x=clientes_sobre_025, color='purple', linestyle='-.', alpha=0.8, linewidth=2,
               label=f'Umbral 0.025 (cliente {clientes_sobre_025:,})')
  
    # Anotación de ganancia máxima
    ax1.annotate(f'Máximo: {ganancia_maxima:,.0f}\nUmbral: {umbral_optimo:.4f}', 
                xy=(indice_maximo, ganancia_maxima),
                xytext=(indice_maximo + len(x_relevante) * 0.15, ganancia_maxima * 1.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='red'))
  
    # Configurar primer gráfico
    ax1.set_xlabel('Clientes ordenados por probabilidad', fontsize=12)
    ax1.set_ylabel('Ganancia Acumulada', fontsize=12)
    titulo = titulo_personalizado or f'Ganancia Acumulada Optimizada - {STUDY_NAME}'
    ax1.set_title(titulo, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
  
    # Segundo gráfico: Distribución de probabilidades
    ax2.hist(y_pred_proba, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    ax2.axvline(x=umbral_optimo, color='red', linestyle='--', linewidth=2, 
               label=f'Umbral Óptimo: {umbral_optimo:.4f}')
    ax2.axvline(x=umbral_025, color='purple', linestyle='-.', linewidth=2, 
               label=f'Umbral 0.025')
  
    ax2.set_xlabel('Probabilidad Predicha', fontsize=12)
    ax2.set_ylabel('Densidad', fontsize=12)
    ax2.set_title('Distribución de Probabilidades Predichas', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
  
    # Ajustar layout
    plt.tight_layout()
  
    # Guardar gráfico con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("resultados", exist_ok=True)
    ruta_archivo = f"resultados/{STUDY_NAME}_ganancia_avanzado_{timestamp}.png"
  
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
  
    # Guardar datos del gráfico en CSV
    ruta_datos = f"resultados/{STUDY_NAME}_datos_ganancia_{timestamp}.csv"
    df_datos = pd.DataFrame({
        'posicion': range(len(ganancias_acumuladas)),
        'ganancia_acumulada': ganancias_acumuladas,
        'probabilidad_ordenada': y_pred_proba[indices_ordenados]
    })
    df_datos.to_csv(ruta_datos, index=False)
  
    logger.info(f"Gráfico avanzado guardado: {ruta_archivo}")
    logger.info(f"Datos guardados: {ruta_datos}")
  
    return ruta_archivo



#-----------------------------> reporte visual completo


def generar_reporte_visual_completo(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   titulo_estudio: str = None) -> dict:
    """
    Genera un reporte visual completo con todos los gráficos y análisis.
  
    Args:
        y_true: Valores verdaderos
        y_pred_proba: Probabilidades predichas
        titulo_estudio: Título personalizado para el estudio
  
    Returns:
        dict: Rutas de todos los archivos generados y estadísticas
    """
    logger.info("=== GENERANDO REPORTE VISUAL COMPLETO ===")
  
    titulo = titulo_estudio or f"Análisis Completo - {STUDY_NAME}"
  
    # 1. Gráfico de ganancia avanzado
    ruta_ganancia = crear_grafico_ganancia_avanzado(y_true, y_pred_proba, titulo)
  
    # 2. Análisis con Polars para estadísticas precisas
    from .gain_function import ganancia_threshold
    analisis_polars = ganancia_threshold(y_true, y_pred_proba)
  
    # 3. Guardar resumen del reporte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_resumen = f"resultados/{STUDY_NAME}_reporte_resumen_{timestamp}.json"
  
    reporte_completo = {
        'metadata': {
            'timestamp': timestamp,
            'titulo_estudio': titulo,
            'total_clientes': len(y_true),
            'distribucion_target': {
                'positivos': int(np.sum(y_true == 1)),
                'negativos': int(np.sum(y_true == 0)),
                'porcentaje_positivos': float(np.mean(y_true) * 100)
            }
        },
        'archivos_generados': {
            'grafico_ganancia': ruta_ganancia,
            'resumen_json': ruta_resumen
        },
        'analisis_polars': analisis_polars,
        'estadisticas_clave': {
            'ganancia_maxima': analisis_polars['ganancia_maxima']['ganancia_maxima'],
            'umbral_optimo': analisis_polars['ganancia_maxima']['umbral_optimo'],
            'clientes_optimos': analisis_polars['ganancia_maxima']['clientes_seleccionados'],
            'mejora_vs_025': analisis_polars['resumen']['mejora_vs_025']
        }
    }
  
    # Guardar resumen en JSON
    import json
    with open(ruta_resumen, 'w') as f:
        json.dump(reporte_completo, f, indent=2, default=str)
  
    logger.info("=== REPORTE VISUAL COMPLETADO ===")
    logger.info(f"Archivos generados:")
    logger.info(f"  - Gráfico ganancia: {ruta_ganancia}")
    logger.info(f"  - Resumen JSON: {ruta_resumen}")
    logger.info(f"Ganancia máxima encontrada: {reporte_completo['estadisticas_clave']['ganancia_maxima']:,.0f}")
  
    return reporte_completo


# -----------------------------------> feature importances

def graficar_importances_test(model, top_n: int = 50):
    """
    Grafica los dos tipos de feature importances del modelo en una sola figura.
   
    Args:
        model: Modelo LightGBM entrenado
        top_n: Número de features más importantes a mostrar
    """
    logger.info("Graficando feature importances (gain, split).")
   
    # Crear figura con 2 subplots horizontales
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
   
    tipos = ['gain', 'split']
    axes = [ax1, ax2]
    colores = ['#2E86AB', '#A23B72']  # Azul, Morado
   
    for ax, tipo, color in zip(axes, tipos, colores):
        # Obtener feature importance
        feature_importance = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(importance_type=tipo)
        }).sort_values('importance', ascending=False).head(top_n)
       
        # Crear gráfico de barras horizontales
        ax.barh(range(len(feature_importance)),
                feature_importance['importance'],
                color=color,
                alpha=0.7,
                edgecolor='black')
       
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['feature'], fontsize=9)
        ax.set_xlabel(f'Importance ({tipo})', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Features by {tipo.upper()}',
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # El más importante arriba
        ax.grid(True, alpha=0.3, axis='x')
       
        # Formatear eje x con separador de miles si es necesario
        if feature_importance['importance'].max() > 1000:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
   
    # Título general
    fig.suptitle(f'Feature Importance Analysis - Test {STUDY_NAME}',
                 fontsize=16, fontweight='bold', y=1.02)
   
    # Ajustar layout
    plt.tight_layout()
   
    # Guardar gráfico con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("resultados", exist_ok=True)
    ruta_archivo = f"resultados/feature_importance_{STUDY_NAME}_{timestamp}.png"
   
    plt.savefig(ruta_archivo, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Gráfico guardado en: {ruta_archivo}")
    plt.close()
   
    # Retornar el DataFrame del tipo 'gain' (el más importante)
    feature_importance_gain = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
   
    return feature_importance_gain