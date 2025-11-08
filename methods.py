import numpy as np

def_tol = 1e-14
def_m = 1000

def newton(g, g_prima, x0, sigma=def_tol, epsilon=def_tol, m=def_m):
    """
    Método de Newton estándar.

    Parámetros:
    g: función.
    g_prima: derivada de g.
    x0: aproximación inicial.
    m: número máximo de iteraciones.
    sigma: toleracia para el error en la variable independiente.
    epsilon: tolerancia para los valores funcionales.

    Retorna:
    Aproximación de una raiz de g en caso de que el método converja.
    """
        
    v = g(x0)
    if np.abs(v) < epsilon:
        return x0
    
    for k in range(m):
        x1 = x0 - v / g_prima(x0)
        v = g(x1)
        if np.abs(x1-x0) < sigma or np.abs(v) < epsilon:
            print(f"Newton terminó tras {k} iteraciones.")
            return x1
        x0 = x1
    print(f"Newton terminó tras con el máxico de iteraciones.")
    return x1


def bisect(g, a, b, sigma=def_tol, epsilon=def_tol, m=def_m):
    """
    Método de bisección estándar.

    Parámetros:
    g: Función.
    a: Extremo izquierdo del intervalo.
    b: Extremo derecho del intervalo.
    m: número máximo de iteraciones.
    sigma: toleracia para el error en la variable independiente.
    epsilon: tolerancia para los valores funcionales.
    Retorna:
    Aproximación de una raiz de g en [a,b], a y b.
    """

    u = g(a)
    v = g(b)
    e = b-a
    if np.sign(u) == np.sign(v):
        raise ValueError(f"Ambos extremos del intervalo tienen igual signo")
    
    for k in range(m):
        e = e/2
        c = a+e
        w = g(c)
        if np.abs(e) < sigma or np.abs(w) < epsilon:
            print(f"Bisección terminó tras {k} iteraciones.")
            return c
        if np.sign(w) != np.sign(u):
            b = c
            v = w
        else:
            a=c
            u=w

    print(f"Bisección terminó tras con el máxico de iteraciones.")

    return c, a, b


def hybrid(g, g_prima, a, b, m, sigma=def_tol, epsilon=def_tol, bisec_max_iterations=10):
    """
    Método híbrido. Realiza a lo sumo bisec_max_iterations iteraciones de bisección y luego
    utiliza Newton a partir del extremo inferior del intervalo. 

    Parámetros:
    g: Función.
    g_prima: derivada de g.
    a: Extremo izquierdo del intervalo.
    b: Extremo derecho del intervalo.
    m: número máximo de iteraciones.
    sigma: toleracia para el error en la variable independiente.
    epsilon: tolerancia para los valores funcionales.
    bisec_max_iterations: número máximo de iteraciones de bisección.
    Retorna:
    Aproximación de una raiz de g en [a,b], a y b.
    """

    # Se requiere que newton inicie a la izquierda de la raiz para asegurar convergencia.
    _, a_1, _ = bisect(g, a, b, sigma, epsilon, bisec_max_iterations)
    
    return newton(g, g_prima, a_1, sigma, epsilon, m)

