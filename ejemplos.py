import numpy as np
from analysis import gather_information
import builders as bl
import miscellaneous as mis
import matplotlib.pyplot as plt

"""
Se proponen dos ejemplos a modo de demostración. 
    * Una matriz primitiva L_1, es decir, con índice de imprimtividad k_1=1, que alcanza una distribución estable en el límite.
    * Una matriz no primitiva L_2, o equivalentemente, con índice de imprimitividad k_2>1, que alterna entre a los sumo k
        configuraciones estables en el límite. 

    Notemos que en realidad ambos casos son descriptos por completo por el primer teorema del informe, solo que en el primer caso, 
    la población "alterna" entre una única configuración estable.

    Adicionalmente, en el segundo caso escalaremos el vector de fecundidades para contemplar los casos lambda_0<1, lambda_0=1 y lambda_0>1. 
"""

# Ejemplo 1 (Matriz basada en el ejercicio 5 del práctico 6 de AN2)
def ejemplo_1():
    # Periodos de un año (simplemente para elegir una unidad)
    f_1 = 2.0*np.array([0.0, 0.2, 0.9, 0.9, 0.9, 0.8, 0.3], dtype=float)
    p_1 = np.array([0.3, 0.7, 0.9, 0.9, 0.9, 0.6], dtype=float)
    X_1 = np.array([10, 2, 8, 5, 12, 0, 1], dtype=float)

    L_1 = bl.build_leslie_matrix(f_1, p_1)

    print("Analicemos la matriz L_1")
    L_1_info = gather_information(f_1, p_1)
    print(L_1_info)

    # Su índice de imprimitividad efectivamente es 1
    # Además, el Teorema 4 nos dice que el factor gamma es positivo para la población inicial X
    # Luego, el Teorema 2 nos indica que la proporción de la población en cada clase converge a
    # Y dado que lambda_0>1, la población debería crecer hasta el infinito
    distribucion_esperada = mis.normalize_1(L_1_info.right_eig)
    print(f"Distribución esperada por franja etaria: {distribucion_esperada}")

    # Se observa que coinciden
    poblacion_obtenida = np.linalg.matrix_power(L_1, 1000) @ X_1
    distribucion_obtenida = mis.normalize_1(poblacion_obtenida)
    print(f"Distribución obtenida por franja etaria tras 1000 periodos (o 1000 años): {distribucion_obtenida}")

    # También podemos expresar la distribución en un gráfico de barras
    mis.display_ages_distributions(np.array([distribucion_esperada]))

    mis.print_splitter()

    # Graficando podemos analizar cómo evoluciona la distribución por etapas y la población total
    # a lo largo del tiempo.
    mis.display_population_evolution(L_1, X_1, 20, L_1_info)

    # Grafiquemos ahora las sensibilidades y las elasticidades. Notemos que efectivamente la suma de las
    # Elasticidades es 1
    print(f"Suma de las elasticidades: {np.sum(L_1_info.elasticities_f) + np.sum(L_1_info.elasticities_p)}")

    mis.print_splitter()

    # Graficamos sensibilidades y elasticidades
    mis.display_population_sen_elast(L_1_info)


# Ejemplo 2 
def ejemplo_2():
    f_2 = 3.0*np.array([0.0, 0.2, 0, 0.9, 0, 0.8], dtype=float)
    p_2 = np.array([0.3, 0.7, 0.9, 0.9, 0.9], dtype=float)
    X_2 = np.array([10, 2, 8, 5, 12, 0], dtype=float)

    L_2 = bl.build_leslie_matrix(f_2, p_2)


    print("Analicemos la matriz L_2")
    L_2_info = gather_information(f_2, p_2)
    print(L_2_info)

    # Su indice de imprimitividad es efectivamente 2
    # Debería alternar entre a lo sumo dos configuraciones estables en el límite. 
    # Verifiquemos esto
    configs = bl.compute_configurations(L_2_info, X_2)
    distribucion_esperada1 = mis.normalize_1(configs[0])
    distribucion_esperada2 = mis.normalize_1(configs[1])
    distribuciones_esperadas = np.apply_along_axis(mis.normalize_1, 1, configs)
    print(f"Distribución esperada por franja etaria")
    print(*distribuciones_esperadas, sep="\n")

    # Se observa que coinciden
    poblacion_obtenida = np.linalg.matrix_power(L_2, 1000) @ X_2
    distribucion_obtenida_1 = mis.normalize_1(poblacion_obtenida)
    distribucion_obtenida_2 = mis.normalize_1(L_2@poblacion_obtenida)
    print("Distribuciones obtenida por franja etaria tras 1000 periodos (o 1000 años):")
    print(f"{distribucion_obtenida_1}")
    print(f"{distribucion_obtenida_2}")


    # También podemos expresar las distribuciones en un gráfico de barras
    mis.display_ages_distributions(distribuciones_esperadas)


    # Graficando podemos analizar cómo evoluciona la distribución por etapas y la población total
    # a lo largo del tiempo.
    mis.display_population_evolution(L_2, X_2, 20, L_2_info)


    # Verifiquemos que el límite predicho por el teorema 1 es correcto para un matriz no primitiva
    mis.display_nonprimitive_evolution(L_2, X_2, 20, configs, L_2_info.lambda_0)


    # Grafiquemos ahora las sensibilidades y las elasticidades. Notemos que efectivamente la suma de las
    # Elasticidades es 1
    print(f"Suma de las elasticidades: {np.sum(L_2_info.elasticities_f) + np.sum(L_2_info.elasticities_p)}")

    mis.print_splitter()

    # Graficamos sensibilidades y elasticidades
    mis.display_population_sen_elast(L_2_info)