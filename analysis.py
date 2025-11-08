import numpy as np
import builders as bl
import matplotlib.pyplot as plt
import methods as met
import testing 

def check_hypothesis(f, p):
    assert np.all(p>0) and np.all(p<=1), "Matriz inválida: Algunas clases serán redundantes en a lo sumo m-1 etapas"
    assert np.any(f>0), "La problación de extinguirá en a lo sumo m etapas"
    assert len(f) == len(p)+1, "Dimensiones inconsistentes"

def display_chacteristic_functions(f, p):
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


def gather_information(f, p):
    m = len(f)

    c = bl.compute_c(p)
    a = bl.compute_a(c,f)
    R = bl.compute_R(a)
    cota_inf, _ = bl.compute_boundaries(R, m)

    q = bl.gen_q(a)
    q_prime = bl.gen_q_prime(a)
    
    lambda_0 = met.newton(q, q_prime, cota_inf)

    right_eig = bl.compute_right_eig(c, lambda_0)
    left_eig = bl.compute_left_eig(c, a, lambda_0)

    sensi_f, sensi_p = bl.compute_sensitivities(left_eig, right_eig)
    elast_f, elast_p = bl.compute_elasticities(f, p, sensi_f, sensi_p, lambda_0)

    print(f"lambda_0: {lambda_0}")
    print(f"right eig: {right_eig}")
    print(f"left eig: {left_eig}")
    print(f"sensi_f: {sensi_f}")
    print(f"sensi_p: {sensi_p}")
    print(f"elast_f: {elast_f}")
    print(f"elast_p: {elast_p}")
    print(f"Sumatoria elasticidades f: {np.sum(elast_f)}")
    print(f"Sumatoria elasticidades p: {np.sum(elast_p)}")
    print(f"Sumatoria elasticidades f y p: {np.sum(elast_f) + np.sum(elast_p)}")

    testing.is_right_eigenpair(f,p, lambda_0, right_eig)
    testing.is_left_eigenpair(f,p, lambda_0, left_eig)


# Matriz de Leslie con tasa de crecimiento mayor a 1
f_1 = np.array([2,3,2,2], dtype=float)
p_1 = np.array([0.9,0.8,0.7], dtype=float)
gather_information(f_1, p_1)
#print(testing.aproximate_lambda_0(f_1, p_1))
#display_chacteristic_functions(f_1, p_1)


# Matriz de Leslie con tasa de crecimiento menor a 1
f_2 = np.array([0.2,0.2,0.1,0.1], dtype=float)
p_2 = np.array([0.9,0.8,0.7], dtype=float)
gather_information(f_2, p_2)
#print(testing.aproximate_lambda_0(f_2, p_2))
#display_chacteristic_functions(f_2, p_2)


# Matriz de Leslie con tasa de crecimiento menor a 1
f_3 = np.array([0, 0, 0.1, 0, 0.3], dtype=float)
p_3 = np.array([0.9,0.8,0.7, 0.9], dtype=float)
gather_information(f_3, p_3)
#print(testing.aproximate_lambda_0(f_3, p_3))
#display_chacteristic_functions(f_3, p_3)

