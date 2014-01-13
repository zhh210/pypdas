'''
Utility classes and functions for generating random matrices, and QP problems.
'''
import random
from cvxopt import matrix, spmatrix, normal, spdiag
from math import pi, sin, cos, pow

def sprandsym(size,cond=100,sp=0.5,vec=None):
    '''
    Generate random sparse positive definite matrix with specified 
    size, cond, sparsity. Implemented by random Jacobi rotation.
    '''
    root = pow(cond,1.0/(size-1))
    if not vec:
        vec = [pow(root,i) for i in range(size)]
    H = spdiag(vec)
    dimension = size*size

    while nnz(H) < sp*dimension*0.95:
        H = rand_jacobi(H)

    # H = normal(size,size)
    # H = H*H.T
    # for i in range(size):
    #     H[i,i] += 20
    #     for j in range(size):
    #         if j != i:
    #             H[i,j] = -abs(H[i,j])
    return H
    
def sp_rand(m,n,a):
     ''' 
     Generates an mxn sparse 'd' matrix with round(a*m*n) nonzeros.
     Provided by cvxopt.
     '''
     if m == 0 or n == 0: return spmatrix([], [], [], (m,n))
     nnz = min(max(0, int(round(a*m*n))), m*n)
     nz = matrix(random.sample(range(m*n), nnz), tc='i')
     J = matrix([k//m for k in nz])
     return spmatrix(normal(nnz,1), nz%m, J, (m,n))

def rand_jacobi(H):
    '''
    Apply random Jacobi rotation on matrix H, preserve eigenvalues, 
    singular values, and symmetry
    '''
    (m,n) = H.size
    theta = random.uniform(-pi,pi)
    c = cos(theta)
    s = sin(theta)
    i = random.randint(0,m-1)
    j = i
    while j == i:
        j = random.randint(0,n-1)

    H[[i,j],:] =  matrix([[c,-s],[s,c]])*H[[i,j],:]
    H[:,[i,j]] =  H[:,[i,j]]*matrix([[c,s],[-s,c]])
    return H

def nnz(H):
    'Compute the number of non-zeros of matrix H'
    num = 0
    for i in H:
        if i != 0:
            num += 1

    return num

def test_rand_jacobi():
    H = sp_rand(5,5,0.5)
    print(H)
    H = rand_jacobi(H)
    print(H)

def test_sprandsym():
    H = sprandsym(6)
    print H

if __name__ == '__main__':
    test_rand_jacobi()
    test_sprandsym()

