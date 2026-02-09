import numpy as np
import math

# QR Decomposition Method (A = QR)

def QR_Decomposition(A):
    A = np.array(A, dtype=float) # Set A as a numpy array with float values
    m, n = A.shape # Get the size of the matrix (row size, column size)
    # Set the Q and R matrix as zero matrices with the corresponding row size and column size
    # Q = orthonormal basis
    # R = upper triangular matrix
    Q = np.zeros((m, n)) 
    R = np.zeros((n, n))

    # For all j (column size)
    for j in range(n):
        # Create a copy of the j th column of A (to prevent real data changes)
        v = A[:, j].copy()

        # For all i up to j 
        for i in range(j):
            q = Q[:, i] # q = i th column of Q
            R[i, j] = np.dot(q, v)  # Calculate the projection
            v -= R[i, j] * q # Subtract the projection from v
        
        # Find the norm of the orthogonal vector (Could be done like this if numpy.linalg functions are allowed -> np.linalg.norm(v)
        R[j, j] = np.sqrt(np.dot(v, v))

        # Normalize the vector (if not zero) and assign it as the j-th column of Q
        if R[j, j] != 0:
            Q[:, j] = v / R[j, j]

    # Return the matrices Q and R 
    return Q.tolist(), R.tolist()


# QR Iteration Mehod
# This method approximates the eigenvalues and eigenvectors of a square matrix A
# In this method we set A(1) = Q(1)R(1)
# Then A(2) = R(1)Q(1) and find A(2) = Q(2)R(2) and so on
# A(k) = Q(k)R(k)
# Eigenvectors will be find using V = Q(1)Q(2)Q(3)...Q(k)

def QR_Iteration(A, iterations=40):
    A = np.array(A, dtype=float) # Set A as a numpy array with float values
    n = A.shape[0] # Get the column size of A
    V = np.identity(n) # Set V as the identity matrix

    # Iterations are determined as 40 according to the experimentations
    for i in range(iterations):
        Q, R = QR_Decomposition(A) # Decompose A into Q and R 
        A = np.dot(R, Q) # Set the next matrix A by multiplying R and Q
        V = np.dot(V, Q) # Multiply V with Q to find the eigenvectors
    
    # Create a list of eigenvalues and add the ones on the diagonal of the matrix A (after 40th iteration)
    eigenvalues = []
    for i in range(n):
        eigenvalues.append(A[i, i])

    # Return the eigenvalues list and the eigenvector matrix V
    return eigenvalues, V.tolist()


# SVD Method using QR Iteration 

def SVD(Matrix):
    A = np.array(Matrix, dtype=float) # Set A as a numpy array with float values
    # Compute A^T A
    At = A.T
    AtA = np.dot(At, A) 

    # Use QR Iteration to compute eigenvalues and eigenvectors of A^T A
    eigenvalues, V = QR_Iteration(AtA)

    # Sort eigenvalues (and corresponding eigenvectors) in descending order
    # In order to keep track of the eigenvalues with its corresponding eigenvector, tuples were created
    # Then it was sorted according to the eigenvalues in descending order
    indexed_eigenvalues = list(enumerate(eigenvalues))
    indexed_eigenvalues.sort(key=lambda x: -x[1])

    # Reorder the eigenvectors to match eigenvectors with the corresponding eigenvalue
    sorted_indices = []
    for pair in indexed_eigenvalues:
        index = pair[0]
        sorted_indices.append(index)

    # Reorder eigenvalues using sorted indices and set negatives to zero
    reordered = []
    for i in sorted_indices:
        value = eigenvalues[i]
        if value > 0:
            reordered.append(value)
        else:
            reordered.append(0)
    eigenvalues = reordered

    # Create singular values list as the square roots of the non-negative eigenvalues
    singular_values = []
    for val in eigenvalues:
        singular_values.append(math.sqrt(val))

    # Reorder the columns of V accordingly
    V = np.array(V)[:, sorted_indices]

    m, n = A.shape # Get the size of the matrix (row size, column size)
    # Construct U matrix using AV / sigma(i)
    U = []
    for i in range(n):
        sigma = singular_values[i] # Get the current sigma
        vi = V[:, i] # Get the current column
        Av = np.dot(A, vi) # Compute A vi
        # Set ui as Av / sigma if sigma is not very small
        if sigma > 1e-10:
            ui = Av / sigma
        # Set ui as zero if sigma is very small
        else:
            ui = np.zeros(m) 
        U.append(ui.tolist())

    # Transpose U to get the correct shape (columns are left singular vectors)
    U = np.array(U).T.tolist()

    # Construct the diagonal matrix E (Σ) with singular values
    # Create a zero matrix E (same shape as U x V^T)
    E = np.zeros((len(U), V.shape[1]))
    # Fill the diagonal of E with the singular values
    # Only go for the smallest number of these three: singular value count, rows of U or columns of V
    for i in range(min(len(singular_values), len(U), V.shape[1])):
        E[i, i] = singular_values[i]

    # Get the transpose of V
    Vt = V.T.tolist()

    # Return all the found matrices (U, E, Vt)
    return U, E, Vt
