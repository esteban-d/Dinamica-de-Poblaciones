import numpy as np
import matplotlib.pyplot as plt
from data import LeslieInformation
from builders import compute_gamma

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,

    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

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

def display_population_evolution(L, X, T, L_1_info: LeslieInformation):
    """
    Muestra dos gráficos: Uno con la evolución del tamaño de la población partiendo de X
    y otro con la evolución de la distribución proporcional por estadios.
    En el primer caso, compara la población total con la predicha por los teoremas
    1 o 3. 
    
    Parámetros:
    X: población inicial.
    L: matriz de Leslie.
    T: períodos.
    L_1_info: Objeto LeslieInformation correspondiente a la matriz de interés.
    """

    m = L.shape[0]

    ts = np.arange(0, T)

    
    ys = np.empty((T, m))
    ys[0] = X


    for i in range(1, T):
        ys[i] = L @ ys[i-1]

    total_ys = np.sum(ys, axis=1)
    ys_norm = ys / np.sum(ys, axis=1, keepdims=True)

    # Por Teorema 2, si la matriz es primitiva, la siguiente función se aproxima en el límite
    # a la problación total.
    # Por Teorema 4, si el índice de imprimitividad es mayor a 1, la función utilizada en el caso primitivo
    # se asemeja en promedio a la del caso primitivo.
    gamma = compute_gamma(L_1_info, X)
    base = gamma * np.sum(L_1_info.right_eig/L_1_info.lambda_0) # Notar que escala autovector (según teorema)
    total_ys_aprox = base * L_1_info.lambda_0**ts


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))

    ax1.plot(ts, total_ys, label="Población total")
    ax1.plot(ts, total_ys_aprox, label="Aproximación - Teorema 2 o 3")
    ax1.set_xlabel("Periodo")
    ax1.set_ylabel("Población")
    ax1.set_title("Evolución de la población total")
    ax1.legend()
    ax1.grid()

    for i in range(m):
        ax2.plot(ts, ys_norm.T[i], label=f"Estadio {i+1}")
    ax2.set_xlabel("Periodo")
    ax2.set_ylabel("Proporción")
    ax2.set_title("Evolución de la distribución por estadio")
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    plt.show()

def display_nonprimitive_evolution(L, X, T, stable_configs, lambda_0):
    """
    Para una matriz no primitiva, grafica el límite expuesto por el Teorema 1 para la población total contra
    la población total real. También puede utilizarse con matrices no primitivas, pero en tal caso el 
    grafico coincidiría con el de display_population_evolution.
    
    Parámetros:
    L: matriz de Leslie.
    X: población inicial.
    T: períodos.
    stable_configs: Vector con las k (indice de imprimitividad) configuraciones estables dadas por el Teorema 1.
    lambda_0: Tasa de crecimiento poblacional
    """

    k = len(stable_configs) # Índice de imprimitividad
    assert k>1

    m = L.shape[0]

    ts = np.arange(0, T)

    ys = np.empty((T, m))
    ys[0] = X
    for i in range(1, T):
        ys[i] = L @ ys[i-1]

    total_ys = np.sum(ys, axis=1)

    # Por el Teorema 1, ys y total_ys_pred deberían igualarse en el límite.
    total_ys_pred = np.empty(T)
    total_population_per_config = np.sum(stable_configs, axis=1)
    for i in range(0, T):
        total_ys_pred[i] = lambda_0**i * total_population_per_config[i%k]

    plt.plot(ts, total_ys, label="Población total")
    plt.plot(ts, total_ys_pred, label="Aproximación - Teorema 1")
    plt.xlabel("Periodo")
    plt.ylabel("Población")
    plt.title("Evolución de la población total")
    plt.legend()
    plt.grid()
    
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
    leslie_information: Objeto LeslieInformation correspondiente a la matriz de interés.
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
    

def display_ages_distributions(distributions):
    """
    Dada una matriz de Leslie, muestra 4 gráficos:
        * Sensibilidades respecto a las tasas de fecundidad.
        * Sensibilidades respecto a las tasas de supervivencia.
        * Elasticidades respecto a las tasas de fecundidad.
        * Elasticidades respecto a las tasas de supervivencia.

    Parámetros:
    leslie_information: Objeto LeslieInformation correspondiente a la matriz de interés.
    """
    k, m = distributions.shape

    rango = np.arange(1,m+1)

    fig, axs = plt.subplots(k,1, figsize=(10, 3*k))
    axs = np.ravel(axs) # De esta forma funciona para matrices primitivas

    for i in range(k):
        axs[i].bar(rango, distributions[i])
        axs[i].set_title(f"Configuración {i}")
        axs[i].set_xlabel("Clase")
        axs[i].set_ylabel("Proporción")

    plt.suptitle("Distribución")
    plt.tight_layout()
    plt.show()