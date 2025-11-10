import numpy as np
import builders as bl
import matplotlib.pyplot as plt
import methods as met
from data import LeslieInformation

def check_hypothesis(f, p):
    """
    Verifica las propiedades asumidas sobre matrices de Leslie bajo las cuales tiene sentido el análisis realizado.

    Parámetros:
    f: arreglo de longitud m con las tasas de fecundidad de cada etapa
    p: arreglo de longitud m-1 con las tasas de supervivencia entre etapas
    """
    assert np.all(p>0) and np.all(p<=1), "Matriz inválida: Algunas clases serán redundantes en a lo sumo m-1 etapas"
    assert np.any(f>0), "La problación de extinguirá en a lo sumo m etapas"
    assert len(f) == len(p)+1, "Dimensiones inconsistentes"

def gather_information(f, p, method="newton"):
    """
    Función principal del proyecto. Calcula la tasa de crecimiento e información adicional que permite
    determinar la evolución de una población.

    Parámetros:
        f: Vector de longitud m con las tasas de fecundidad de cada etapa.
        p: Vector de longitud m-1 con las tasas de superviviencia entre etapas

    Retorna:
        LeslieInformation, un objeto con la información calculada, la cual se encuentra detallada en la
        documentación de data.py.
    """
    check_hypothesis(f,p)

    m = len(f)

    c = bl.compute_c(p)
    a = bl.compute_a(c,f)
    R = bl.compute_R(a)
    cota_inf, cota_sup = bl.compute_boundaries(R, m)

    q = bl.gen_q(a)
    q_prime = bl.gen_q_prime(a)
    if method == "newton":
        lambda_0 = met.newton(q, q_prime, cota_inf)
    elif method == "bisec":
        lambda_0 = met.bisect(q, cota_inf, cota_sup)
    else:
        raise Exception("Método Inválido")
     
    g_i_lambda_0 = bl.compute_g_i(a, lambda_0)

    right_eig = bl.compute_right_eig(c, lambda_0)
    left_eig = bl.compute_left_eig(c, g_i_lambda_0, lambda_0)

    sensi_f, sensi_p = bl.compute_sensitivities(left_eig, right_eig)
    elast_f, elast_p = bl.compute_elasticities(f, p, sensi_f, sensi_p, lambda_0)

    imprimitivity_index = bl.compute_imprimitivity_index(f)

    poly_p = bl.compute_poly_p(a)

    return LeslieInformation(lambda_0, right_eig,left_eig, sensi_f, sensi_p, elast_f, elast_p,imprimitivity_index, a, c, R, poly_p, g_i_lambda_0)


def display_chacteristic_functions(f, p):
    """
    Esta función tiene como objetivo visualizar las funciones caracteríticas de una matríz de Leslie (mxm). 
    Puede resultar útil en casos extremos para ayudar a elegir el mejor método para encontrar la tasa de crecimiento poblacional.
    Grafica las funciones p(x) y q(x), las cotas superiores e inferiores de lambda_0, y aproximaciones de lambda_0 mediante
    bisección y newton.

    Parámetros:
    f: arreglo de longitud m con las tasas de fecundidad de cada etapa
    p: arreglo de longitud m-1 con las tasas de supervivencia entre etapas
    """
    check_hypothesis(f,p)

    m = len(f)
    c = bl.compute_c(p)
    a = bl.compute_a(c,f)
    R = bl.compute_R(a)
    cota_inf, cota_sup = bl.compute_boundaries(R, m)
    poly_p = bl.compute_poly_p(a)
    q = bl.gen_q(a)
    q_prime = bl.gen_q_prime(a)

    lambda_0_bisec = met.bisect(q, cota_inf, cota_sup)
    lambda_0_newton = met.newton(q, q_prime, cota_inf)

    lambdas = np.linspace(cota_inf, cota_sup, int(np.ceil(10000*(cota_sup-cota_inf))))[1:]
    p_vals = poly_p(lambdas)

    q_vals = np.array([bl.compute_q_for(a, x) for x in lambdas])

    plt.plot(lambdas, p_vals, label=fr"$p(\lambda)$")
    plt.plot(lambdas, q_vals, label=fr"$q(\lambda)$")
    plt.scatter(cota_inf, 0, color="red", s=60, label=fr'$\lambda_0$ (cota inferior) = {cota_inf}')
    plt.scatter(cota_sup, 0, color="green", s=60, label=fr'$\lambda_0$ (cota superior) = {cota_sup}')
    plt.scatter(lambda_0_bisec, 0, color="orange", s=20, label=fr'$\lambda_0$ (bisección) = {lambda_0_bisec}')
    plt.scatter(lambda_0_newton, 0, color="purple", s=20, label=fr'$\lambda_0$ (newton) = {lambda_0_newton}')
    plt.xlabel(fr"$\lambda$")
    plt.ylabel("Valor Función)")
    plt.legend()
    plt.title(fr"Funciones con raiz $\lambda_0$")
    plt.grid()
    plt.show()


