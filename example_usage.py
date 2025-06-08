from gauss_seidel_solver import gauss_seidel_method
import numpy as np

if __name__ == "__main__":
    # Example 1
    A1 = np.array([[5, 2, 1],
                   [-1, 4, 2],
                   [2, -3, 10]])
    b1 = np.array([7, 3, -1])
    print("--- Example 1 ---")
    try:
        solution1, num_iterations1 = gauss_seidel_method(A1, b1, verbose=True)
        print("\nSolution vector:", solution1)
        print("Number of iterations:", num_iterations1)
    except ValueError as e:
        print(f"Error: {e}")
    except TypeError as e: # Add TypeError handling
        print(f"Error: {e}")

    print("\n" + "="*30 + "\n")

    # Example 2 (Another system)
    A2 = np.array([[10., -1., 2., 0.],
                   [-1., 11., -1., 3.],
                   [2., -1., 10., -1.],
                   [0., 3., -1., 8.]])
    b2 = np.array([6., 25., -11., 15.])
    print("--- Example 2 ---")
    try:
        solution2, num_iterations2 = gauss_seidel_method(A2, b2, verbose=True)
        print("\nSolution vector:", solution2)
        print("Number of iterations:", num_iterations2)
    except ValueError as e:
        print(f"Error: {e}")
    except TypeError as e: # Add TypeError handling
        print(f"Error: {e}")

    print("\n" + "="*30 + "\n")

    # Example 3: Non-square matrix (will raise ValueError)
    print("--- Example 3: Non-square matrix ---")
    A3 = np.array([[1, 2], [3, 4], [5, 6]])
    b3 = np.array([1, 2, 3])
    try:
        gauss_seidel_method(A3, b3, verbose=True)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except TypeError as e:
        print(f"Caught unexpected error: {e}")

    print("\n" + "="*30 + "\n")

    # Example 4: Zero diagonal element (will raise ValueError)
    print("--- Example 4: Zero diagonal element ---")
    A4 = np.array([[1, 2], [0, 4]])
    b4 = np.array([1, 2])
    try:
        gauss_seidel_method(A4, b4, verbose=True)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except TypeError as e:
        print(f"Caught unexpected error: {e}")
