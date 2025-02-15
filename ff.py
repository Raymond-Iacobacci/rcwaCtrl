import ast

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import numpy.typing as npt

import pandas as pd  # type: ignore

def quar(mat: npt.ArrayLike) -> np.ndarray:
    assert mat.shape[0] == mat.shape[1]
    assert mat.shape[0] % 2 == 0
    mid = int(mat.shape[0] / 2)
    return np.array([mat[:mid, :mid], mat[:mid, mid:], mat[mid:, :mid], mat[mid:, mid:]])

# TODO vectorize for GPU use and give errors
def manual_matmul(A, B, threshold=None, dtype=None):
    if dtype is None:
        dtype = np.cdouble
    else:
        dtype = np.dtype(dtype)
    """
    Manually performs matrix multiplication between two NumPy arrays A and B.

    Parameters:
    - A: NumPy array of shape (m, n)
    - B: NumPy array of shape (n, p)

    Returns:
    - result: NumPy array of shape (m, p) resulting from A x B
    """

    # Get the dimensions of the input matrices
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape

    # Check if the matrices can be multiplied
    if a_cols != b_rows:
        raise ValueError("Incompatible dimensions for matrix multiplication.")

    # Initialize the result matrix with zeros
    # This class has worked so far to elim half-zero errors
    result = np.zeros((a_rows, b_cols), dtype=dtype)

    # Perform the matrix multiplication manually
    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(a_cols):  # or range(b_rows)
                result[i, j] += A[i, k] * B[k, j]
    if threshold is not None:
        result[np.abs(result) < threshold] *= 0
    return result

def read_boolean_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        _, boolean_values = line.split(",", 1)
        boolean_list = ast.literal_eval(boolean_values.strip())
        int_list = [int(value) for value in boolean_list]
        data.append(int_list)
    data_array = np.array(data, dtype=int)
    return data_array


def parse_data(lines):
    data_arrays = []
    for line in lines:
        if line.strip():  # This checks if the line is not empty
            array_str = line.split(",", 1)[1].strip().strip("[]")
            array = np.array(list(map(float, array_str.split(","))))
            data_arrays.append(array)
    return np.array(data_arrays)


def read_data(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            value = line.strip().split(", ")[1]
            data.append(float(value))
    return np.array(data, dtype=object)


def plotLens(array):
    plt.figure(figsize=(10, 5))
    plt.step(range(len(array)), array, where="post")
    plt.show()

def printdf(ipt: np.ndarray) -> None:
    np.set_printoptions(linewidth=120, precision=6, suppress=True)
    df = pd.DataFrame(ipt)
    manual_format = "\n".join(
        " ".join(f"{val.real:+.6f}{val.imag:+.6f}j" for val in row) for row in ipt)
    print(df)
    
# def custom_eigendecomposition(X, max_iter=1000, tol=1e-12):
#     """
#     Compute the eigendecomposition of a complex, non-symmetric matrix X
#     from scratch using a basic QR-iteration approach (without shifts).
    
#     Returns
#     -------
#     V : ndarray of shape (n, n)
#         The eigenvector matrix.
#     D : ndarray of shape (n, n)
#         The diagonal eigenvalue matrix (same order as used internally).
#     res_decomp : ndarray of shape (n, n)
#         The residual X - V @ D @ V^-1.
#     res_evals : ndarray of shape (n, n)
#         The residual X - D @ I, where I is the n x n identity.
        
#     Notes
#     -----
#     1. This is a demonstration of the unshifted QR algorithm. It may converge
#        slowly or fail to converge for certain matrices.
#     2. We do not use np.linalg.eig or scipy.linalg.eig, but do use basic
#        operations like np.linalg.inv for convenience.
#     3. The order of the eigenvalues in D (and corresponding columns in V)
#        is determined by the order in which the QR iteration produces
#        the (quasi-)triangular form.
#     4. For large or tricky matrices, consider more robust methods with shifts,
#        deflation strategies, etc.
#     """
#     # Ensure X is a NumPy array of type complex (to avoid issues)
#     A = np.array(X, dtype=np.complex128)
#     n, m = A.shape
#     if n != m:
#         raise ValueError("Matrix must be square.")

#     # 1) Convert A to (upper) Hessenberg form via Householder reflections.
#     #    This step speeds up QR iteration and is standard. 
#     H, Q_hess = _to_hessenberg(A)

#     # 2) Perform iterative (unshifted) QR on the Hessenberg form.
#     #    We accumulate the transformations (Q_k's) to eventually get eigenvectors.
#     H_final, Q_total = _qr_iteration(H, max_iter=max_iter, tol=tol)

#     # 3) At this point, H_final is (in principle) upper triangular if everything converged,
#     #    and the diagonal entries of H_final are the eigenvalues in the internal order.
#     #    The columns of V = Q_hess @ Q_total are the eigenvectors in that same order.
#     V = Q_hess @ Q_total
    
#     # 4) Extract the eigenvalues from the diagonal of H_final
#     eigenvals = np.diag(H_final)
    
#     # Build the diagonal eigenvalue matrix D, preserving the exact order:
#     D = np.diag(eigenvals)

#     # 5) Compute the residuals:
#     #    res_decomp = X - V @ D @ V^-1
#     #    res_evals  = X - D @ I
#     V_inv = np.linalg.inv(V)
#     res_decomp = A - V @ D @ V_inv
    
#     # For X - D @ I, just do A - D:
#     # Because D is n x n, I is n x n => D @ I = D
#     I_n = np.eye(n, dtype=np.complex128)
#     res_evals = A - D @ I_n

#     return V, D, res_decomp, res_evals


# def _to_hessenberg(A):
#     """
#     Reduce A to upper Hessenberg form using Householder transformations.
#     Returns H, Q so that A = Q @ H @ Q^H   (Q^H is the conjugate transpose),
#     where H is upper Hessenberg and Q is unitary.
#     """
#     A = A.copy()
#     n = A.shape[0]
#     Q_total = np.eye(n, dtype=np.complex128)

#     for k in range(n-2):
#         # 1) Construct the Householder vector to zero out subdiagonal entries below A[k, k].
#         x = A[k+1:, k]
#         # Norm of x
#         norm_x = np.linalg.norm(x)
#         if norm_x < 1e-15:
#             continue
        
#         # Householder vector v
#         v = x.copy()
#         # We'll flip the sign of x[0] to create the reflection
#         v[0] += np.sign(x[0]) * norm_x if x[0] != 0 else norm_x
#         # Normalize v
#         v = v / np.linalg.norm(v)
        
#         # 2) Apply the transformation to A from left and right (in practice, we only need left on A).
#         # Build the reflection block: I - 2 v v^H
#         # but note that v is (n-k-1) long, so we embed into an n x n matrix for the operation.
#         H_k = np.eye(n, dtype=np.complex128)
#         H_k_sub = np.eye(n-k-1, dtype=np.complex128) - 2.0 * np.outer(v, v.conjugate())
#         H_k[k+1:, k+1:] = H_k_sub
        
#         A = H_k @ A @ H_k.conjugate().T
#         Q_total = Q_total @ H_k.conjugate().T

#     return A, Q_total


# def _qr_iteration(H, max_iter=1000, tol=1e-12):
#     """
#     Perform unshifted QR iteration on upper Hessenberg matrix H.
#     Returns R, Qacc so that R ~ Qacc^H * H * Qacc, with R being (nearly) upper triangular.
#     Qacc accumulates all Q_i from each QR factorization step, i.e.
#         Qacc = Q_1 * Q_2 * ... * Q_m
#     """
#     n = H.shape[0]
#     Qacc = np.eye(n, dtype=np.complex128)
#     R = H.copy()

#     for _ in range(max_iter):
#         # Check if R is already (close to) upper triangular
#         off_diag_norm = np.linalg.norm(np.tril(R, -1))
#         if off_diag_norm < tol:
#             break
        
#         # Basic unshifted QR: factor R = Q * U
#         Q, U = _qr_decomposition(R)
#         R = U @ Q
#         Qacc = Qacc @ Q

#     return R, Qacc


# def _qr_decomposition(M):
#     """
#     Compute the (complex) QR factorization of M via Householder reflections:
#         M = Q * R
#     Returns Q, R.
#     """
#     M = M.copy()
#     n = M.shape[0]
#     Q = np.eye(n, dtype=np.complex128)
    
#     for k in range(n):
#         # Create Householder vector to eliminate below diagonal in column k
#         x = M[k:, k]
#         norm_x = np.linalg.norm(x)
#         if norm_x < 1e-15:
#             continue
        
#         v = x.copy()
#         v[0] += np.sign(x[0]) * norm_x if x[0] != 0 else norm_x
#         v = v / np.linalg.norm(v)
        
#         # Apply reflection to M from the left
#         H_k = np.eye(n, dtype=np.complex128)
#         H_sub = np.eye(n-k, dtype=np.complex128) - 2.0 * np.outer(v, v.conjugate())
#         H_k[k:, k:] = H_sub
        
#         M = H_k @ M
#         Q = Q @ H_k.conjugate().T

#     return Q, M  # Q, R






def qr_decomposition(A):
    """
    Perform QR decomposition of matrix A using Gram-Schmidt orthogonalization.
    Returns matrices Q and R such that A = Q @ R
    """
    n, m = A.shape
    Q = np.zeros((n, m), dtype = complex)
    R = np.zeros((m, m), dtype = complex)
    
    for j in range(m):
        v = A[:, j].copy()
        
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            raise ValueError("Matrix is rank deficient")
        Q[:, j] = v / R[j, j]
    
    return Q, R

def is_converged(A, tol):
    """
    Check if the matrix A has converged to an upper triangular matrix within the given tolerance.
    Convergence is achieved when all off-diagonal elements are below tol.
    """
    off_diagonal = A - np.diag(np.diagonal(A))
    return np.all(np.abs(off_diagonal) < tol)

def eigendecomposition_qr(M, max_iterations=1000, tol=1e-10):
    """
    Perform eigendecomposition of matrix M using the QR algorithm without sorting eigenvalues.
    Returns a list of eigenvalues in their natural order as they emerge from the QR iterations.
    
    Parameters:
    - M: Square numpy matrix
    - max_iterations: Maximum number of QR iterations
    - tol: Tolerance for convergence
    
    Returns:
    - eigenvalues: List of eigenvalues in natural order
    """
    A = M.copy().astype(complex)
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square.")
    
    for it in range(max_iterations):
        Q, R = qr_decomposition(A)
        A_next = R @ Q
        
        if is_converged(A_next, tol):
            A = A_next
            print(f"Converged in {it+1} iterations.")
            break
        
        # else:
        #     print("QR algorithm did not converge within the maximum number of iterations.")
    
    # Extract eigenvalues from the diagonal
    eigenvalues = np.diagonal(A)
    
    # Handle possible complex eigenvalues
    # Since QR for real matrices may not capture complex eigenvalues, additional handling is needed
    # Here, we assume real eigenvalues for simplicity
    return eigenvalues

def find_eigenvectors(M, eigenvalues, tol=1e-8):
    """
    Compute eigenvectors corresponding to the provided eigenvalues.
    
    Parameters:
    - M: Square numpy matrix
    - eigenvalues: List or array of eigenvalues
    - tol: Tolerance for zero
    
    Returns:
    - eigenvectors: List of eigenvectors corresponding to each eigenvalue
    """
    n = M.shape[0]
    eigenvectors = []
    
    for idx, lambda_val in enumerate(eigenvalues):
        # Solve (M - lambda * I) v = 0
        A = M - lambda_val * np.eye(n)
        # Use Gaussian elimination to find the null space
        # Here, we'll use SVD for numerical stability
        U, S, Vh = np.linalg.svd(A)
        # The eigenvector is the last column of V (or row of Vh)
        v = Vh[-1, :]
        # Normalize the eigenvector
        v_normalized = v / np.linalg.norm(v)
        eigenvectors.append(v_normalized)
    
    return eigenvectors