import numpy as np
from numpy.polynomial import Polynomial

def compute_c(p):
    """
    Calcula las probabilidades acumuladas de superviviencia c_i entre cualesquiera dos estadíos.
    
    Parámetros: 
    p: Vector de longitud m-1 tal que p[i] es la tasa de supervivencia de la etapa i+1 a la 
    etapa i+2.

    Retorna:
    c: Vector de longitud m tal que: 
        c[0] = 1 
        c[k] = p[0] * p[1] * ... * p[k-1], para k=1,...,m-1
    """
    m = len(p)+1
    c = np.empty(m)
    c[0] = 1

    for i in range(1,m):
        c[i] = c[i-1] * p[i-1]

    return c



def compute_a(c, f):
    """
    Calcula el producto de Hadarmard entre las probabilidades acumuladas de superviviencia y
    las tasas de fecundidad.

    Parámetros:
    c: Vector de longitud m con las probabilidades acumuladas de superviviencia.
    f: Vector de longitud m con las tasas de fecundidad.

    Retorna:
    a: Producto de Hadamard de c y f
    """

    return c*f    


def compute_R(a):
    """
    Calcula el valor reproductivo neto.

    Parámetros:
    a: Vector de longitud m resultante del producto de Hadamard de las probabilidades acumuladas de superviviencia y
    las tasas de fecundidad.

    Retorna:
    R: Valor reproductivo neto de la población
    """
    return np.sum(a)

def compute_boundaries(R, m):
    """
    Calcula una cota inferior positiva y una superior, para el único autovalor positivo lambda_0, es decir
    la tasa de crecimiento poblacional.

    Parámetros:
    R: Valor reproductivo neto de la población.
    m: Número de estadíos poblacionales.

    Retorna:
    Vector de longitud dos tal que su primer elemento es una cota superior de la tasa de crecimiento poblacional
    y su segundo elemento es una cota inferior positiva de la tasa de crecimiento poblacional     
    """
    if R < 1:
        return R, R**(1/m)
    
    return R**(1/m), R


def compute_poly_p(a):
    """
    Calcula el polinomio característico (salvo tal vez por un signo) de la matriz de Leslie.

    Parámetros:
    a: Vector de longitud m resultante del producto de Hadamard de las probabilidades acumuladas de superviviencia y
    las tasas de fecundidad.

    Retorna:
    Polinomio tal que sus raíces coinciden con los autovalores de la matriz de Leslie.
    """
    
    # Coeficientes [1, -a[0],...,-a[m-1]]
    coeficients = np.append(-np.flip(a), 1)
    
    return Polynomial(coeficients)


def compute_q(a, x):
    """
    Devuelve q(x), donde "q" es la función definida en el informe si x es escalar. Si es un vector, devuelve un vector
    con la función evaluada en cada punto.

    Parámetros:
    a: Vector de longitud m resultante del producto de Hadamard de las probabilidades acumuladas de superviviencia y
    las tasas de fecundidad.
    x: Escalar o vector de longitud l en el cual se evalua la función.

    Retorna:
    q(x) o un arreglo de la forma [q(x[0]),...,q(x[l])].
    
    """
    m=len(a)

    # Caso escalar:
    if x.ndim == 0:
        powers = x ** -np.arange(1, m+1)
        return 1 - np.sum(a * powers)

    # Caso vector:
    powers = x[:,None]**-np.arange(1, m+1)  # Análogo a un producto externo
    return 1 - np.sum(a * powers, axis=1)   # Sumar filas


def compute_right_eig(c, lambda_0):
    """
    Calcula un autovector derecho con entradas no negativas asociado al único autovalor positivo de la matriz de Leslie
    (o tasa de crecimiento poblacional).

    Parámetros:
    c: Vector de longitud m con las probabilidades acumuladas de superviviencia.
    lambda_0: tasa de crecimiento poblacional.

    Retorna:
    Autovector derecho sociado a lambda_0 tal que su primera entrada es 1.
    """

    m = len(c)
    x = lambda_0 ** np.arange(m)
    return c/x


def aux_left_eig(a, lambda_0):
    """
    Función auxiliar de compute_left_eig. Calcula cada g_i(lambda_0), donde g_i es la función definida en el informe. 

    Parámetros:
    a: Vector de longitud m resultante del producto de Hadamard de las probabilidades acumuladas de superviviencia y
    las tasas de fecundidad.
    lambda_0: tasa de crecimiento poblacional.

    Retorna:
    Vector v de tamaño m tal que v[i] = g_{i+1}(lambda_0), para i=0,...,m-1
    """
    m = len(a)

    g_lambda_0 = np.empty(m)
    g_lambda_0[-1] = a[-1]  # Equivalente a m-1

    # De esta forma se consigue un algoritmo con complejidad lineal 
    for j in reversed(range(m-1)):
        g_lambda_0[j] = g_lambda_0[j+1] + a[j] * lambda_0**(m-j)
    return g_lambda_0


def compute_left_eig(c, a, lambda_0):
    """
    Calcula un autovector izquierdo con entradas no negativas asociado al único autovalor positivo de la matriz de Leslie
    (o tasa de crecimiento poblacional).

    Parámetros:
    c: Vector de longitud m con las probabilidades acumuladas de superviviencia.
    a: Vector de longitud m resultante del producto de Hadamard de las probabilidades acumuladas de superviviencia y
    las tasas de fecundidad.
    lambda_0: tasa de crecimiento poblacional.

    Retorna:
    Autovector izquierdo sociado a lambda_0 tal que su primera entrada es 1.
    """
    m = len(c)

    g = aux_left_eig(a, lambda_0)

    powers = lambda_0 ** np.arange(m,0,-1)

    return g / (c * powers)

    
