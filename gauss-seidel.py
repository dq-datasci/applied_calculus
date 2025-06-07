import numpy as np

def gauss_seidel_method(A, b, initial_guess=None, tolerance=1e-6, max_iterations=100):
    """
    Gauss-Seidel Iterative Method for solving a system of linear equations Ax = b.

    Parameters:
    - A: Coefficient matrix.
    - b: Right-hand side vector.
    - initial_guess: Initial guess for the solution vector (default is None).
    - tolerance: Tolerance for stopping criteria.
    - max_iterations: Maximum number of iterations.

    Returns:
    - x: Solution vector.
    - iterations: Number of iterations performed.
    """
    
    A = A.astype(np.float64)
    b = b.astype(np.float64)

    n = len(b)

    x = initial_guess if initial_guess is not None else np.zeros(n)
    error = np.zeros(n)
    
    y = [chr(i) for i in range(97, 97+n)]
    print(f'it: {y} Error:')

    for iteration in range(max_iterations):
        print(f'{iteration}: {x} {error}')
        x_new = np.copy(x)
        for i in range(n):
            sigma = sum(A[i, j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i, i]

        if np.allclose(x, x_new, rtol=tolerance):
            return x_new, iteration + 1
    
        error = abs(x - x_new)

        x = x_new

    raise ValueError("Gauss-Seidel method did not converge within the specified number of iterations.")

# Example usage:
A = np.array([[5, 2, 1],
              [-1, 4, 2],
              [2, -3, 10]])

b = np.array([7, 3, -1])

solution, num_iterations = gauss_seidel_method(A, b)
print("Solution vector:", solution)
print("Number of iterations:", num_iterations)