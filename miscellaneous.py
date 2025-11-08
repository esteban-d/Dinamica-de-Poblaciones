import numpy as np
import matplotlib.pyplot as plt
from data import LeslieInformation

def normalize_1(v):
    """
    Normaliza un vector en norma 1
    """
    return v/np.linalg.norm(v, 1)

def print_splitter(l=143):
    """
    Imprime un separador
    """
    print("-"*l)

def display_population_evolution(L, X, T):
    """
    Muestra dos gráficos: Uno con la evolución del tamaño de la población patiendo de X
    y otro con la evolución de la distribución proporcional por estadíos de la población.
    
    Parámetros:
    X: población inicial
    L: matriz de Leslie
    T: períodos
    """

    m = L.shape[0]

    ts = np.arange(0, T)
    ys = np.empty((T, m))
    ys[0] = X


    for i in range(1, T):
        ys[i] = L @ ys[i-1]

    total_ys = np.sum(ys, axis=1)
    ys_norm = ys / np.sum(ys, axis=1, keepdims=True)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))

    ax1.plot(ts, total_ys, label="Evolución de la población total")
    ax1.set_xlabel("Periodo")
    ax1.set_ylabel("Población Total")
    ax1.set_title("Población Total")
    ax1.legend()
    ax1.grid()

    for i in range(m):
        ax2.plot(ts, ys_norm.T[i], label=f"estadío {i+1}")
    ax2.set_xlabel("Periodo")
    ax2.set_ylabel("Proporción")
    ax2.set_title("Evolución de la distrubución por estadío")
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    plt.show()


def display_population_sen_elast(leslie_information: LeslieInformation):
    """
    Dada una matriz de Leslie, muestra 4 gráficos:
        * Sensibilidades respecto a las tasas de fecundidad.
        * Sensibilidades respecto a las tasas de supervivencia.
        * Elasticidades respecto a las tasas de fecundidad.
        * Elasticidades respecto a las tasas de supervivencia.

    Parámetros:
    leslie_information:
    Objeto LeslieInformation correspondiente a la matriz de interés.
    """


    sensi_f = leslie_information.sensitivities_f
    sensi_p = leslie_information.sensitivities_p
    elast_f = leslie_information.elasticities_f
    elast_p = leslie_information.elasticities_p
    m = len(sensi_f)

    rango_f = np.arange(1,m+1)
    rango_p = np.arange(1,m)

    color_sensi = "orange"
    color_elast = "blue"

    fig, axs = plt.subplots(2,2, figsize=(17,10))
    
    axs[0,0].bar(rango_f, sensi_f, color=color_sensi)
    axs[0,0].set_title("Sensibilidades - Fecundidad")
    axs[0,0].set_xlabel("Clase")
    axs[0,0].set_ylabel("Sensibilidad")

    axs[0,1].bar(rango_f, elast_f, color=color_elast)
    axs[0,1].set_title("Elasticidades - Fecundidad")
    axs[0,1].set_xlabel("Clase")
    axs[0,1].set_ylabel("Elasticidad")

    axs[1,0].bar(rango_p, sensi_p, color=color_sensi)
    axs[1,0].set_title("Sensibilidades - Supervivencia")
    axs[1,0].set_xlabel("Clase")
    axs[1,0].set_ylabel("Sensibilidad")

    axs[1,1].bar(rango_p, elast_p, color=color_elast)
    axs[1,1].set_title("Elasticidad - Supervivencia")
    axs[1,1].set_xlabel("Clase")
    axs[1,1].set_ylabel("Elasticidad")

    plt.tight_layout()
    plt.show()
    


def display_age_distribution(leslie_information: LeslieInformation):
    """
    Representa una distribución etaria estable en un gráfico de barras

    Parámetros:
    leslie_information:
    Objeto LeslieInformation correspondiente a la matriz de interés.
    """
    assert leslie_information.imprimitivity_index==1, "Pueden haber múltiples distribuciones estables"

    dist = normalize_1(leslie_information.left_eig)
    m = len(dist)
    
    plt.bar(np.arange(1,m+1), dist, color="green")
    plt.title("Distribución estable")
    plt.xlabel("Clase")
    plt.ylabel("Proporción")

    plt.tight_layout()
    plt.show()
