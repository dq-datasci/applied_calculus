from gauss_seidel_solver import gauss_seidel_method
import numpy as np
from colorama import Fore, Back, Style, init

init(autoreset=True)

if __name__ == "__main__":
    # Example 1
    A1 = np.array([[5, 2, 1],
                   [-1, 4, 2],
                   [2, -3, 10]])
    b1 = np.array([7, 3, -1])
    print("--- Example 1 ---")
    # Simplemente llama a la función, el decorador se encarga del resto
    gauss_seidel_method(A1, b1, verbose=True)

    # Example 2 (Another system)
    A2 = np.array([[10., -1., 2., 0.],
                   [-1., 11., -1., 3.],
                   [2., -1., 10., -1.],
                   [0., 3., -1., 8.]])
    b2 = np.array([6., 25., -11., 15.])
    print("--- Example 2 ---")
    gauss_seidel_method(A2, b2, verbose=True)

    # Example 3: Non-square matrix (will raise ValueError)
    print("--- Example 3: Non-square matrix ---")
    A3 = np.array([[1, 2], [3, 4], [5, 6]])
    b3 = np.array([1, 2, 3])
    gauss_seidel_method(A3, b3, verbose=True)

    # Example 4: Zero diagonal element (will raise ValueError)
    print("--- Example 4: Zero diagonal element ---")
    A4 = np.array([[0, 2], 
                   [3, 4]])
    b4 = np.array([1, 2])
    gauss_seidel_method(A4, b4, verbose=True)

    # Example 5: Dimensions of matrix A and vector b do not match.(will raise ValueError)
    print("--- Example 5: Dimensions of matrix A and vector b do not match ---")
    A5 = np.array([[3, 2], 
                   [3, 4]])
    b5 = np.array([1, 2, 3])
    gauss_seidel_method(A5, b5, verbose=True)