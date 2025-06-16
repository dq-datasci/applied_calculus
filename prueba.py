from gauss_seidel_solver import gauss_seidel_method
import numpy as np

if __name__ == "__main__":
    # Example 1
    A1 = np.array([[3, -1, 1],
                   [3, 6, 2],
                   [3, 3, 7]])
    b1 = np.array([1, 0, 4])
    print("--- Example 1 ---")
    # Simplemente llama a la función, el decorador se encarga del resto
    gauss_seidel_method(A1, b1, verbose=True)