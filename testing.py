import numpy as np
import builders as bl
from analysis import gather_information, display_chacteristic_functions


def aproximate_lambda_0(f, p):
    L = bl.build_leslie_matrix(f, p)
    eigenvalues = np.linalg.eig(L)[0]

    for eigenvalue in eigenvalues:
        if np.isreal(eigenvalue) and eigenvalue>0:
            return eigenvalue
        

def is_right_eigenpair(f, p, lambda_0, v):
    L = bl.build_leslie_matrix(f, p)

    assert np.allclose(L@v, lambda_0*v)


def is_left_eigenpair(f, p, lambda_0, v):
    L = bl.build_leslie_matrix(f, p)

    assert np.allclose(v@L, lambda_0*v)


# Matriz de Leslie con tasa de crecimiento mayor a 1
X_1 = np.array([7,10,9,10], dtype=float)
f_1 = np.array([2,3,2,0], dtype=float)
p_1 = np.array([0.9,0.8,0.7], dtype=float)

info_1 = gather_information(f_1, p_1).forPopulation(X_1)
#print(testing.aproximate_lambda_0(f_1, p_1))
#display_chacteristic_functions(f_1, p_1)


# Matriz de Leslie con tasa de crecimiento menor a 1
X_2 = np.array([7,10,9,10], dtype=float)
f_2 = np.array([0.2,0.2,0.1,0.1], dtype=float)
p_2 = np.array([0.9,0.8,0.7], dtype=float)
info_2 = gather_information(f_2, p_2).forPopulation(X_2)
#print(testing.aproximate_lambda_0(f_2, p_2))
#display_chacteristic_functions(f_2, p_2)


# Matriz de Leslie con tasa de crecimiento menor a 1
X_3 = np.array([7,10,9,10,10], dtype=float)
f_3 = np.array([0, 0, 0.1, 0, 0.3], dtype=float)
p_3 = np.array([0.9,0.8,0.7, 0.9], dtype=float)
info_3 = gather_information(f_3, p_3).forPopulation(X_3)
#print(testing.aproximate_lambda_0(f_3, p_3))
#display_chacteristic_functions(f_3, p_3)

# Matriz de Leslie con tasa de crecimiento menor a 1. Caso extremo.
X_4 = np.array([7,10,9,10,10], dtype=float)
f_4 = np.array([0, 0, 0, 0, 0.1], dtype=float)
p_4 = np.array([0.9,0.8,0.7, 0.9], dtype=float)
info_4 = gather_information(f_4, p_4).forPopulation(X_4)
#print(testing.aproximate_lambda_0(f_3, p_3))
#display_chacteristic_functions(f_4, p_4)

