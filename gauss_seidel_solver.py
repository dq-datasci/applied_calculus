import numpy as np
from colorama import Fore, Style, Back, init
import functools 

init(autoreset=True) # Para que los colores se reseteen automáticamente

# --- Definición del Decorador ---
def handle_gauss_seidel_execution(func):
    """
    Un decorador que envuelve la ejecución de la función Gauss-Seidel,
    agrega separadores visuales y maneja excepciones específicas.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Aquí es donde se llama a la función original (gauss_seidel_method)
            solution, num_iterations = func(*args, **kwargs)
            # exito() se llama si la función se ejecuta sin errores
            exito(solution, num_iterations) 
            return solution, num_iterations
        except ValueError as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            return None, None
        except TypeError as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            return None, None
        except Exception as e: # Captura cualquier otra excepción inesperada
            print(f"{Fore.RED}Ocurrió un error inesperado: {e}{Style.RESET_ALL}")
            return None, None
        finally:
            print("\n" + "="*70 + "\n") # Separador inferior
    return wrapper

# --- Aplicar el Decorador a gauss_seidel_method ---
@handle_gauss_seidel_execution
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
        raise TypeError(f"{Fore.RED}{Style.BRIGHT}Error de Tipo:{Style.RESET_ALL} A y b deben ser arrays de NumPy.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}Error de Validación:{Style.RESET_ALL} La matriz A debe ser cuadrada.")
    if A.shape[0] != len(b):
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}Error de Validación:{Style.RESET_ALL} Las dimensiones de la matriz A y el vector b no coinciden.")

    A = A.astype(np.float64)
    b = b.astype(np.float64)

    n = len(b)
    
    # Check for zero diagonal elements
    if np.any(np.diag(A) == 0):
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}Error de Configuración:{Style.RESET_ALL} Elemento(s) diagonal(es) de la matriz A son cero. Gauss-Seidel puede no converger o causar división por cero.")

    x = initial_guess if initial_guess is not None else np.zeros(n, dtype=np.float64) 
 
    
    if verbose:
        header_vars = [f'x_{i}' for i in range(n)]
        print(f"Iteración: {Fore.CYAN}{Style.DIM}{' '.join(header_vars):<30} {Fore.LIGHTMAGENTA_EX}{Style.DIM}Error (Norma L2):{Style.RESET_ALL}")
    for iteration in range(max_iterations):
        x_old = np.copy(x) # Store the old x for error calculation
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]

        # Calculate error using the L2 norm for a single scalar value
        current_error = np.linalg.norm(x - x_old) 
        
        if verbose:
            print(f"{iteration:<9}: {Style.DIM}{Fore.LIGHTCYAN_EX}{str(np.round(x, 6)):<30}{Style.RESET_ALL} {Style.DIM}{Fore.MAGENTA}{current_error:.6e}{Style.RESET_ALL}")
        if current_error < tolerance:
            return x, iteration + 1
        
    raise ValueError(f"{Fore.RED}{Style.BRIGHT}Error de Convergencia:{Style.RESET_ALL} El método de Gauss-Seidel no convergió dentro de {max_iterations} iteraciones. "
                     f"Error actual: {current_error:.6e}")

def exito(sol_vec, num_iter, title="¡Convergencia exitosa!"):
    print("\n" + Back.GREEN + Fore.WHITE + Style.BRIGHT + f" {title.upper()} ")
    print("-" * (len(title) + 4))
    print(f"\nSolution vector: {Fore.GREEN}{sol_vec}")
    print(f"Number of iterations: {Fore.GREEN}{num_iter}")