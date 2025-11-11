import numpy as np
from numpy.polynomial import Polynomial
from data import LeslieInformation

def compute_c(p):
    """
    Calcula las probabilidades acumuladas de superviviencia c_i.
    
    Parámetros: 
    p: Vector de longitud m-1 tal que p[i] es la tasa de supervivencia de la etapa i+1 a la 
    etapa i+2 (considerando a 1 como la primera etapa).

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
    m: Número de estadios poblacionales.

    Retorna:
    Vector de longitud dos tal que su primer elemento es una cota inferior positiva de la tasa de crecimiento poblacional
    y su segundo elemento es una cota superior de la tasa de crecimiento poblacional     
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


def compute_q_for(a, x):
    """
    Devuelve q(x), donde "q" es la función definida en el informe si x es escalar. Si es un vector, devuelve un vector
    con la función evaluada en cada punto.

    Parámetros:
    a: Vector de longitud m resultante del producto de Hadamard de las probabilidades acumuladas de superviviencia y
    las tasas de fecundidad.
    x: Escalar o vector de longitud l en el cual se evalua la función.

    Retorna:
    q(x) o un arreglo de la forma [q(x[0]),...,q(x[l])], respectivamente.
    """
    m=len(a)

    # Caso escalar:
    if x.ndim == 0:
        powers = x ** -np.arange(1, m+1)
        return 1 - np.sum(a * powers)

    # Caso vector:
    powers = x[:,None]**-np.arange(1, m+1)  # Análogo a un producto externo
    return 1 - np.sum(a * powers, axis=1)   # Sumar filas


def compute_q_prime_for(a, x):
    """
    Devuelve q'(x), donde "q" es la función definida en el informe si x es escalar. Si es un vector, devuelve un vector
    con la función evaluada en cada punto.

    Parámetros:
    a: Vector de longitud m resultante del producto de Hadamard de las probabilidades acumuladas de superviviencia y
    las tasas de fecundidad.
    x: Escalar o vector de longitud l en el cual se evalua la función.

    Retorna:
    q'(x) o un arreglo de la forma [q'(x[0]),...,q'(x[l])], respectivamente.
    """

    m=len(a)

    # Caso escalar:
    if x.ndim == 0:
        powers = x ** -np.arange(2, m+2)
        powers = powers * np.arange(1,m+1)
        return np.sum(a * powers)

    # Caso vector:
    powers = x[:,None]**-np.arange(2, m+2)  # Análogo a un producto externo
    coefs = a * np.arange(1,m+1)
    return np.sum(coefs * powers, axis=1)   # Sumar filas


def gen_q(a):
    """
    Devuelve q, donde "q" es la función definida en el informe.

    Parámetros:
    a: Vector de longitud m resultante del producto de Hadamard de las probabilidades acumuladas de superviviencia y
    las tasas de fecundidad.

    Retorna:
    q.
    """

    m = len(a)
    exp = -np.arange(1, m+1)
    def q(x):
        return 1 - np.sum(a * (x ** exp))
    
    return q

def gen_q_prime(a):
    """
    Devuelve q', donde "q" es la función definida en el informe.

    Parámetros:
    a: Vector de longitud m resultante del producto de Hadamard de las probabilidades acumuladas de superviviencia y
    las tasas de fecundidad.

    Retorna:
    q'.
    """
    m=len(a)
    exp = -np.arange(2, m+2)
    coefs = a * np.arange(1,m+1)

    # Caso escalar:
    def q_prime(x):
        return np.sum(coefs * (x ** exp))

    return q_prime


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


def compute_g_i(a, lambda_0):
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
        g_lambda_0[j] = g_lambda_0[j+1] + a[j] * lambda_0**(m-j-1)

    return g_lambda_0


def compute_left_eig(c, g_i_lambda_0, lambda_0):
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

    powers = lambda_0 ** np.arange(m,0,-1)

    return g_i_lambda_0 / (c * powers)


def compute_sensitivities(v, w):
    """
    Calcula la sensibilidad de lambda_0 respecto a cada parámetro de la matriz de Leslie mxm.

    Parámetros:
    v: autovector izquierdo asociado a lambda_0.
    w: autovector derecho asociado a lambda_0.

    Retorna:
    sensi_f: vector de longitud m tal que sensi_f[i] es la sensibilidad de lambda_0 respecto a 
    la tasa de fertilidad de la (i+1)-ésima etapa (contando las etapas a partir de 1).
    sensi_s: vector de longitud m-1 tal que sensi_p[i] es la sensibilidad de lambda_0 respecto a la
    tasa de superviviencia de la (i+1)-ésima etapa a la (i+2)-ésima etapa.
    """

    m = len(v)
    # Sensibilidades de las tasas de fecundidad
    sensi_f = np.empty(m)
    # Sensibilidad de las tasas de superviviencia
    sensi_p = np.empty(m-1)

    for i in range(m):
        sensi_f[i] = (v[0]*w[i]) / np.inner(v, w)

    for i in range(m-1):
        sensi_p[i] = (v[i+1]*w[i]) / np.inner(v, w)

    return sensi_f, sensi_p

def compute_elasticities(f, p, sensi_f, sensi_p, lambda_0):
    """
    Calcula la elasticidad de lambda_0 respecto a cada parámetro de la matriz de Leslie mxm.

    Parámetros:
    v: autovector izquierdo asociado a lambda_0.
    w: autovector derecho asociado a lambda_0.
    sensi_f: vector de longitud m con las sensibilidades de lambda_0 respecto a las tasas de fertilidad.
    sensi_p: vector de longitud m-1 con las sensibilidades de lambda_0 respecto a las tasas de supervivencia.
    lambda_0: tasa de crecimiento poblacional.

    Retorna:
    elast_f: vector de longitud m tal que elast_f[i] es la elasticidad de lambda_0 respecto a 
    la tasa de fertilidad de la (i+1)-ésima etapa (contando las etapas a partir de 1).
    elast_s: vector de longitud m-1 tal que elast_p[i] es la elasticidad de lambda_0 respecto a la
    tasa de superviviencia de la (i+1)-ésima etapa a la (i+2)-ésima etapa.
    """

    m = len(f)
    # Elasticidades de las tasas de fecundidad
    elast_f = np.empty(m)
    # Elasticidades de las tasas de superviviencia
    elast_p = np.empty(m-1)

    for i in range(m):
        elast_f[i] = f[i] * sensi_f[i] / lambda_0

    for i in range(m-1):
        elast_p[i] = p[i] * sensi_p[i] / lambda_0

    return elast_f, elast_p


def build_leslie_matrix(f, p):
    """
    Construye una matriz de Leslie mxm.

    Parámetros:
    f: Vector de largo m con las tasas de fecundidad.
    p: Vector de largo m-i con las tasas de superviviencia entre etapas.

    Retorna: 
    L: Matriz de Leslie mxm construida a partir de f y p.
    """

    n = len(f)
    L = np.zeros((n,n))

    L[0,:] = f
    L[1:, 0:n-1] = np.diag(p[:n-1])

    return L


def compute_imprimitivity_index(f):
    """
    Calcula el índice de imprimitividad de una matriz de Leslie L (mxm).

    Parámetros:
    f: Vector de largo m con las tasas de fecundidad de L.

    Retorna: 
    Índice de imprimitividad de L.
    """

    indexes = np.where(f>0)[0] + 1 
    return np.gcd.reduce(indexes)

def is_viable(information: LeslieInformation, X):
    # También podría usarse f
    a = information.a
    if np.any(X!=0):
        i_X = np.min(np.where(X!=0)[0])
        i_a = np.max(np.where(a!=0)[0])
        return i_X <= i_a
    
    return False



def compute_avg_eigen(information: LeslieInformation, X):
    """
    Calcula el autovector definido en el Teorema 2, aunque también utilizado en el
    Teorema 3, es decir, el autovector gamma*[(c_1/lambda_0), ... , (c_i/(lambda_0^i)) , ... , (c_m/(lambda_0^m))], 
    o un vector nulo si gamma es nulo.

    Parámetros:
    leslie_information: Objeto LeslieInformation correspondiente a la matriz de interés.
    X: Vector con la población inicial.

    Retorna
    v: Autovector definido en el Teorema 2.
    """
    c = information.c
    m = len(c)

    # Teorema 4 permite identificar si gamma es nulo sin errores numéricos
    if not is_viable(information, X):
        return np.zeros(m, dtype=float)

    p_prime = information.poly_p.deriv()
    lambda_0 = information.lambda_0
    g_j_lambda_0 = information.g_i_lambda_0

    gamma = 0
    for j in range(m):
        gamma += ( (lambda_0 ** j) * g_j_lambda_0[j] * X[j]) / (p_prime(lambda_0) * c[j])

    v = information.right_eig / lambda_0

    return gamma*v


def compute_configurations(information: LeslieInformation, x):
    """
    Para una matriz de Leslie con índice de imprimitividad k, calcula las (a lo sumo) k configuraciones
    entre las cuales alterna definidas por el Teorema 1. En particular, calcula la única configuración
    definida en el Teorema 2 si k=1.

    Parámetros:
    leslie_information: Objeto LeslieInformation correspondiente a la matriz de interés.
    X: Vector con la población inicial.

    Retorna:
    Vector con las k definidas por el Teorema 1.
    """
    p_prime = information.poly_p.deriv()
    c = information.c
    lambda_0 = information.lambda_0
    g_j_lambda_0 = information.g_i_lambda_0
    m = len(c)

    p_prime_lambda_0 = p_prime(lambda_0)
    k = information.imprimitivity_index

    right_eigen = information.right_eig

    configurations = np.full((k, m), right_eigen/lambda_0)

    for t in range(k):
        for i in range(1, m+1):
            gamma = 0
            for j in range(1, m+1):
                if j % k == (i-t) % k:
                    gamma = gamma + (lambda_0**(j-1) * g_j_lambda_0[j-1] * x[j-1]) / c[j-1]
            configurations[t, i-1] = configurations[t, i-1] * gamma * k / p_prime_lambda_0

    return configurations
