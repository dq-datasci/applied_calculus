import numpy as np

def gauss_seidel_method(A, b, initial_guess=None, tolerance=1e-6, max_iterations=100, verbose=False):
    """
    Gauss-Seidel Iterative Method for solving a system of linear equations Ax = b.

    Parameters:
    - A: Coefficient matrix (numpy.ndarray).
    - b: Right-hand side vector (numpy.ndarray).
    - initial_guess: Initial guess for the solution vector (numpy.ndarray, default is None).
    - tolerance: Relative tolerance for stopping criteria (float).
    - max_iterations: Maximum number of iterations (int).
    - verbose: If True, prints iteration details (bool).

    Returns:
    - x: Solution vector (numpy.ndarray).
    - iterations: Number of iterations performed (int).

    Raises:
    - TypeError: If A or b are not numpy arrays.
    - ValueError: If A is not a square matrix, dimensions of A and b do not match,
                  a diagonal element of A is zero, or the method does not converge.
    """

    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("A and b must be numpy arrays.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be a square matrix.")
    if A.shape[0] != len(b):
        raise ValueError("Dimensions of matrix A and vector b do not match.")

    A = A.astype(np.float64)
    b = b.astype(np.float64)

    n = len(b)
    
    # Check for zero diagonal elements
    if np.any(np.diag(A) == 0):
        raise ValueError("Diagonal element(s) of matrix A are zero. Gauss-Seidel may not converge or divide by zero.")

    x = initial_guess if initial_guess is not None else np.zeros(n, dtype=np.float64) 
 
    
    if verbose:
        header_vars = [f'x_{i}' for i in range(n)]
        print(f"Iteration: {' '.join(header_vars):<30} Error (L2 Norm):")

    for iteration in range(max_iterations):
        x_old = np.copy(x) # Store the old x for error calculation
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]

        # Calculate error using the L2 norm for a single scalar value
        current_error = np.linalg.norm(x - x_old) 
        
        if verbose:
            print(f"{iteration:<9}: {str(np.round(x, 6)):<30} {current_error:.6e}")

        if current_error < tolerance:
            return x, iteration + 1
        
    raise ValueError(f"Gauss-Seidel method did not converge within {max_iterations} iterations. "
                     f"Current error: {current_error:.6e}")

