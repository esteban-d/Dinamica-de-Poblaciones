from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial
from dataclasses import asdict


@dataclass
class LeslieInformation:
    """
    Contiene información sobre una matriz de Leslie L de dimensiones mxm.

    Atributos:
        lambda_0: Tasa de crecimiento poblacional, es decir, el único
        autovalor positivo de L.
        right_eig: Autovector derecho asociado a lambda_0 (distribución estable) tal que su primera entrada es 1.
        left_eig: Autovector izquierdo asociado a lambda_0 (valores reproductivos).
        sensitivities_f: Vector de longitud m con las sensibilidades de lambda_0 respecto a cada tasa de fecundidad.
        sensitivities_p: Vector de longitud m-1 con las sensibilidades de lambda_0 respecto a cada tasa de supervivencia.
        elasticities_f: Vector de longitud m con las elasticidades de lambda_0 respecto a cada tasa de fecundidad.
        elasticities_p: Vector de longitud m-1 con las elasticidades de lambda_0 respecto a cada tasa de supervivencia.
        f: Vector de longitud m con las tasas de fertilidad.
        p: Vector de longitud m-1 con las tasas de superviviencia entre etapas.
        imprimitivity_index: Índice de imprimitividad de L.
        a: Producto de Hadamard entre los vectores fecundidad y supervivencia acumulada.
        c: Vector de longitud m con la supervivencia acumulada.
        R: Valor reproductivo neto.
        poly_p: Polinomio tal que sus raíces son los autovalores de la matriz (característico salvo quizás por el signo).
        g_i_lambda_0: Funciones g_i definidas en el informe evaluadas en lambda_0.
    """
    lambda_0: float
    right_eig: np.ndarray 
    left_eig: np.ndarray 
    sensitivities_f: np.ndarray
    sensitivities_p: np.ndarray
    elasticities_f: np.ndarray
    elasticities_p: np.ndarray
    f: np.ndarray
    p: np.ndarray
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
            f"Información principal de la matriz de Leslie:\n"
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
    
    def forPopulation(self, X) -> 'PopulationInformation':
        """
        Computa las configuraciones definidas por el teorema 1 y el autovector definido en el teorema 2.

        Parámetros:
        X: Vector con la población inicial.

        Retorna:
        PopulationInformation, un objeto con la información asociada a la matriz de Leslie original y a la población,
        la cual se encuentra detallada en la documentación de data.py.
        """
        from builders import compute_avg_eigen, compute_configurations

        # Autovector definido en el teorema 2
        avg_eigen = compute_avg_eigen(self, X)

        # Configuraciones definidas por el teorema 1
        configurations = compute_configurations(self, X)

        return PopulationInformation(
            **asdict(self), 
            X=X,
            avg_eigen=avg_eigen,
            configs=configurations,
        )

@dataclass
class PopulationInformation(LeslieInformation):
    """
    Contiene información sobre una matriz de Leslie L de dimensiones mxm pertinente a LeslieInformation
    junto a información extra asociada a una población inicial.

    Atributos:
        X: Vector de longitud m con la población inicial.
        avg_eiven: Vector de longitud m con el autovector asociado a la tasa de crecimiento definido en el teorema 2
        o un vector nulo si el gamma fuese nulo.
        configs: Matriz kxm tal que sus filas contienen las configuraciones dadas por el teorema 1.
    """

    X: np.ndarray
    avg_eigen: np.ndarray
    configs: np.ndarray

    def __repr__(self):
        leslie_info = super().__repr__()
        return(
            leslie_info +
            f"Información principal sobre la población\n"
            f"-----------------------------------------------------------------------------------------------------------------------------------------------\n"
            f"Población inicial: {self.X}\n"
            f"Autovector (teorema 2 o 3): {self.avg_eigen}\n"
            f"Configuraciones: {self.configs}\n"
            f"-----------------------------------------------------------------------------------------------------------------------------------------------\n"
            f"Predicción\n"
            f"-----------------------------------------------------------------------------------------------------------------------------------------------\n"
            f"{self._diagnose_growth()}"
            f"-----------------------------------------------------------------------------------------------------------------------------------------------\n"
        )

    def _diagnose_growth(self):
        tol_eq = 1e-7

        if self.lambda_0<1-tol_eq or np.allclose(self.configs, 0):
            return "La población se extinguirá.\n"

        if self.imprimitivity_index>1:
            oscilatorio_str = f"La población alternará entre las siguientes {self.imprimitivity_index} configuraciones:"
        else:
            oscilatorio_str = f"La población alcanzará la configuración estable:"

        configs_norm_1 = self.configs / np.linalg.norm(self.configs, ord=1, axis=1, keepdims=True)

        lines = []
        for i, config in enumerate(configs_norm_1):
            lines.append(f"Configuración {i}: {np.round(config, 4)}.")
        configs_str = "\n".join(lines)

        behaviour_str = "Además, la población total "
        if np.abs(self.lambda_0-1) < 1e-7:
            behaviour_str += "se mantendrá estable"
        else:
            behaviour_str += "crecerá sin límite"

        if self.imprimitivity_index > 1:
            behaviour_str += " en promedio"
        behaviour_str+="."

        return (
            f"{oscilatorio_str}\n"
            f"{configs_str}\n"
            f"{behaviour_str}\n"
        )
