import numpy as np
import builders as bl


def aproximate_lambda_0(f, p):
    L = bl.build_leslie_matrix(f, p)
    eigenvalues = np.linalg.eig(L)[0]

    for eigenvalue in eigenvalues:
        if np.isreal(eigenvalue) and eigenvalue>0:
            return eigenvalue
        

def is_right_eigenpair(f, p, lambda_0, v):
    L = bl.build_leslie_matrix(f, p)
    print(f"L@v={L@v}")
    print(f"lambda_0*v={lambda_0*v}")
    assert np.allclose(L@v, lambda_0*v)


def is_left_eigenpair(f, p, lambda_0, v):
    L = bl.build_leslie_matrix(f, p)

    print(f"v@L = {v@L}")
    print(f"lambda_0*v = {lambda_0*v}")
    assert np.allclose(v@L, lambda_0*v)

