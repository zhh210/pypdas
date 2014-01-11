from cvxopt import matrix, spmatrix, spdiag
import numpy as np

def test_set_operations():
    pass

def test_filter_opertations():
    # Compare vector value, a < b
    a = matrix(range(5))
    b = matrix([np.Inf]*5)

def test_mat_mult():
    a = matrix(range(16),(4,4))
    b = spmatrix(range(4),range(4),range(4),(4,4))
    c = spdiag(range(4))
    print a*b
    print a*c
