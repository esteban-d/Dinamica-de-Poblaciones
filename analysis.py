import numpy as np
import builders as bl
import matplotlib.pyplot as plt

def check_hypothesis(f, p):
    assert np.all(p>0) and np.all(p<=1), "Matriz inválida: Algunas clases serán redundantes en a lo sumo m-1 etapas"
    assert np.any(f>0), "La problación de extinguirá en a lo sumo m etapas"
    assert len(f) == len(p)+1, "Dimensiones inconsistentes"

def display_functions(f, p):
    m = len(f)
    c = bl.compute_c(p)
    a = bl.compute_a(c,f)
    R = bl.compute_R(a)
    cota_inf, cota_sup = bl.compute_boundaries(R, m)

    poly_p = bl.compute_poly_p(a)


    lambdas = np.linspace(cota_inf, cota_sup, int(np.ceil(10000*(cota_sup-cota_inf))))[1:]
    p_vals = poly_p(lambdas)

    q_vals = np.array([bl.compute_q(a, x) for x in lambdas])

    plt.plot(lambdas, p_vals, label=fr"$p(\lambda)$")
    plt.plot(lambdas, q_vals, label=fr"$q(\lambda)$")
    plt.scatter(cota_inf, 0, color="red", s=60, label=fr'$\lambda_0$ (cota inferior) $= {cota_inf}$')
    plt.scatter(cota_sup, 0, color="red", s=60, label=fr'$\lambda_0$ (cota superior $= {cota_sup}$')
    plt.xlabel("lambda")
    plt.ylabel("Valor Función)")
    plt.legend()
    plt.title("Funciones con raiz lambda_0")
    plt.grid()
    plt.show()

# Matriz de supervivencia
# Ma
f = np.array([2,3,2,2], dtype=float)
p = np.array([0.9,0.8,0.7], dtype=float)
display_functions(f, p)