import numpy as np
import builders as bl
from analysis import gather_information


def approximate_lambda_0(f, p):
    """Calcula el único autovalor real positivo utilizando NumPy."""

    L = bl.build_leslie_matrix(f, p)
    eigenvalues = np.linalg.eigvals(L)

    positive_eigenvalues = [
        eigenvalue.real
        for eigenvalue in eigenvalues
        if np.isclose(eigenvalue.imag, 0) and eigenvalue.real > 0
    ]

    if len(positive_eigenvalues) != 1:
        raise ValueError(
            "Se esperaba encontrar exactamente un autovalor real positivo."
        )

    return positive_eigenvalues[0]


def is_right_eigenpair(f, p, lambda_0, v):
    L = bl.build_leslie_matrix(f, p)
    return np.allclose(L @ v, lambda_0 * v)


def is_left_eigenpair(f, p, lambda_0, v):
    L = bl.build_leslie_matrix(f, p)
    return np.allclose(v @ L, lambda_0 * v)


def check_case(name, X, f, p, should_grow):
    """Verifica los resultados principales para uno de los casos."""

    info = gather_information(f, p).forPopulation(X)
    lambda_0_numpy = approximate_lambda_0(f, p)

    assert np.isclose(info.lambda_0, lambda_0_numpy)
    assert is_right_eigenpair(f, p, info.lambda_0, info.right_eig)
    assert is_left_eigenpair(f, p, info.lambda_0, info.left_eig)
    assert np.isclose(
        np.sum(info.elasticities_f) + np.sum(info.elasticities_p),
        1,
    )

    if should_grow:
        assert info.lambda_0 > 1
    else:
        assert info.lambda_0 < 1

    print(f"{name}: correcto")
    print(f"Tasa de crecimiento: {info.lambda_0}")
    print()


# Matriz de Leslie con tasa de crecimiento mayor a 1
X_1 = np.array([7, 10, 9, 10], dtype=float)
f_1 = np.array([2, 3, 2, 0], dtype=float)
p_1 = np.array([0.9, 0.8, 0.7], dtype=float)


# Matriz de Leslie con tasa de crecimiento menor a 1
X_2 = np.array([7, 10, 9, 10], dtype=float)
f_2 = np.array([0.2, 0.2, 0.1, 0.1], dtype=float)
p_2 = np.array([0.9, 0.8, 0.7], dtype=float)


# Matriz de Leslie con tasa de crecimiento menor a 1
X_3 = np.array([7, 10, 9, 10, 10], dtype=float)
f_3 = np.array([0, 0, 0.1, 0, 0.3], dtype=float)
p_3 = np.array([0.9, 0.8, 0.7, 0.9], dtype=float)


# Matriz de Leslie con tasa de crecimiento menor a 1. Caso extremo.
X_4 = np.array([7, 10, 9, 10, 10], dtype=float)
f_4 = np.array([0, 0, 0, 0, 0.1], dtype=float)
p_4 = np.array([0.9, 0.8, 0.7, 0.9], dtype=float)


def check_all_cases():
    check_case("Caso 1", X_1, f_1, p_1, should_grow=True)
    check_case("Caso 2", X_2, f_2, p_2, should_grow=False)
    check_case("Caso 3", X_3, f_3, p_3, should_grow=False)
    check_case("Caso 4", X_4, f_4, p_4, should_grow=False)


if __name__ == "__main__":
    check_all_cases()


# Para inspeccionar gráficamente alguno de los casos:
# display_chacteristic_functions(f_1, p_1)