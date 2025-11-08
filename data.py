from dataclasses import dataclass
import numpy as np


@dataclass
class LeslieInformation:
    """
    Contiene información sobre una matriz de Leslie L de dimensiones mxm.

    Atributos:
        lambda_0: Tasa de crecimiento poblacional, es decir, el único
        autovalor positivo de L.
        right_eig: Autovector derecho asociado a lambda_0 (distribución estable) tal que su primera entrada es 1.
        left_eig: Autovector izquierdo asociado a lambda_0 (valores reproductivos)
        sensitivities_f: Vector de longitud m con las sensibilidades de lambda_0 respecto a cada tasa de fecundidad.
        sensitivities_p: Vector de longitud m-1 con las sensibilidades de lambda_0 respecto a cada tasa de supervivencia.
        elasticities_f: Vector de longitud m con las elasticidades de lambda_0 respecto a cada tasa de fecundidad.
        elasticities_p: Vector de longitud m-1 con las elasticidades de lambda_0 respecto a cada tasa de supervivencia.
        imprimitivity_index: Índice de imprimitividad de L.
        a: Producto de Hadamard entre los vectores fecundidad y supervivencia acumulada.
        c: Vector de longitud m con la supervivencia acumulada.
        R: Valor reproductivo neto.
    """

    lambda_0: float
    right_eig: np.ndarray 
    left_eig: np.ndarray 
    sensitivities_f: np.ndarray
    sensitivities_p: np.ndarray
    elasticities_f: np.ndarray
    elasticities_p: np.ndarray
    imprimitivity_index: int
    a: np.ndarray
    c: np.ndarray
    R: np.ndarray

