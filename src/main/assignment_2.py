# Question 1
# Nevilles method
import numpy as np

def neville_method(x, y, w):
    n = len(x)
    neville = np.zeros((n, n))
    
    # Initialize first column with y values
    for i in range(n):
        neville[i][0] = y[i]
    
    # Compute Neville's table
    for j in range(1, n):
        for i in range(n - j):
            term1 = (w - x[i + j]) * neville[i][j - 1]
            term2 = (w - x[i]) * neville[i + 1][j - 1]
            neville[i][j] = (term1 - term2) / (x[i] - x[i + j])
    
    return neville[0][n - 1]

# Question 2
# Newtons forward method
def newton_forward(x, y):
    n = len(x)
    diffs = np.zeros((n, n))
    
    # Initialize first column with y values
    for i in range(n):
        diffs[i][0] = y[i]
    
    # Compute forward differences
    for j in range(1, n):
        for i in range(n - j):
            diffs[i][j] = (diffs[i + 1][j - 1] - diffs[i][j - 1]) / (x[i + j] - x[i])
    
    return diffs

# Question 3
# f(7.3) approximation
def newton_forward_interpolation(x, diffs, w):
    n = len(x)
    result = diffs[0][0]
    term = 1.0
    for i in range(1, n):
        term *= (w - x[i - 1])
        result += term * diffs[0][i]
    return result

# Question 4 
# Hermite polynomial
def hermite_interpolation(x, fx, dfx):
    n = len(x)
    col_limit = 2 * n - 1  # Dynamically set column limit
    H = np.zeros((2 * n, col_limit))  # Matrix with variable columns
    z = np.zeros(2 * n)
    
    for i in range(n):
        z[2 * i] = x[i]
        z[2 * i + 1] = x[i]
        H[2 * i][0] = fx[i]
        H[2 * i + 1][0] = fx[i]
        H[2 * i + 1][1] = dfx[i]
        if i != 0:
            H[2 * i][1] = (H[2 * i][0] - H[2 * i - 1][0]) / (z[2 * i] - z[2 * i - 1])
    
    for j in range(2, col_limit):  # Use dynamic column limit
        for i in range(j, 2 * n):
            H[i][j] = (H[i][j - 1] - H[i - 1][j - 1]) / (z[i] - z[i - j])
    
    return z, H, col_limit

# Question 5
# Cubic spline
def cubic_spline_matrix(x, fx):
    n = len(x)
    A = np.zeros((n, n))
    b = np.zeros(n)
    h = np.diff(x)
    
    A[0, 0] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((fx[i + 1] - fx[i]) / h[i] - (fx[i] - fx[i - 1]) / h[i - 1])
    A[n - 1, n - 1] = 1
    
    x_vector = np.linalg.solve(A, b)
    return A, b, x_vector