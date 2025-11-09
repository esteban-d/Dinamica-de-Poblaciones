from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial

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
        poly_p: Polinomio tal que sus raíces son los autovalores de la matriz (característico salvo quizás por el signo).
        g_i_lambda_0: Funciones g_i definidas en el informe evaluadas en lambda_0
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
    poly_p: Polynomial
    g_i_lambda_0: np.ndarray

    def __repr__(self):
        """
        Imprime la información más relevante para el análisis:
        """
        return(
            f"-----------------------------------------------------------------------------------------------------------------------------------------------\n"
            f"Información de la matriz de Leslie:\n"
            f"-----------------------------------------------------------------------------------------------------------------------------------------------\n"
            f"Tasa de crecimiento (autovalor dominante): {self.lambda_0}\n"
            f"Autovector derecho: {self.right_eig}\n"
            f"Autovector izquierdo: {self.left_eig}\n"
            f"Sensibilidades respecto a tasas de fecundidad: {self.sensitivities_f}\n"
            f"Sensibilidades respecto a tasas de superviviencia: {self.sensitivities_p}\n"
            f"Elasticidades respecto a tasas de fecundidad: {self.elasticities_f}\n"
            f"Elasticidades respecto a tasas de superviviencia: {self.elasticities_p}\n"
            f"Índice de imprimitividad: {self.imprimitivity_index}\n"
            f"Valor reproductivo neto: {self.R}\n"
            f"-----------------------------------------------------------------------------------------------------------------------------------------------\n"
        )

