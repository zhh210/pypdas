'''
An example demonstrating solving a descritized optimal boundary control problem.
'''
from cvxopt import spmatrix, sparse, spdiag, matrix
from numpy import inf
from math import sin, cos
from copy import copy

from pdas.pdas import PDAS
from prob import QP
from randutil import sp_rand

import pdb
import sys

def Laplace(n,dic):
    'Create discretized Laplace operator (in 2 dimensions). Modified'
    # Initialize the matrix by a sparse matrix
    L = spmatrix([],[],[],(0,n**2+4*n))

    h = 1.0/(n-1)
    # Decide the coefficient matrix
    for i in range(0,n):
        for j in range(0,n):
            index = dic[(i,j)]
            y = spmatrix([],[],[],(1,n**2+4*n))
            y[0,index]  = -4 + h**2
            y[0,dic[(i-1,j)]] = 1
            y[0,dic[(i+1,j)]] = 1
            y[0,dic[(i,j-1)]] = 1
            y[0,dic[(i,j+1)]] = 1
            L = sparse([L,y])
    return L

def geteq(n,dic):
    'generate the equality constraints.'
    # Equality from the Laplace operator
    AeqL = -Laplace(n,dic)
    h = 1.0/(n-1)


    # Equality from the boundary
    AeqB = spmatrix([],[],[],(0,n**2+4*n))
    for col in range(n):
        y = spmatrix([],[],[],(2,n**2+4*n))
        y[0,dic[-1,col]] = 1
        y[0,dic[1,col]] = -1

        y[1,dic[n-2,col]] = -1
        y[1,dic[n,col]] = 1
        AeqB = sparse([AeqB,y])

    for row in range(n):
        y = spmatrix([],[],[],(2,n**2+4*n))
        y[0,dic[row,-1]] = 1
        y[0,dic[row,1]] = -1

        y[1,dic[row,n-2]] = -1
        y[1,dic[row,n]] = 1
        AeqB = sparse([AeqB,y])
    
    AeqL = sparse([[AeqL],[spmatrix([],[],[],(n**2,n*4))]])
    AeqB = sparse([[AeqB],[-identity(4*n,h*2)]])

    Aeq = sparse([AeqL,AeqB])
    beq = spmatrix([],[],[],(n**2+4*n,1))
    return (Aeq, beq)

def getlu(n,phi):
    'generate the variational upper bound.'
    l = matrix(-inf,(n**2+8*n,1))
    u = matrix([matrix(inf,(n**2+4*n,1)), matrix(phi,(n*4,1))])
    return (l,u)

def getH(n,beta):
    'generate the Hessian.'
    y = identity(n**2+4*n)
    u = identity(n*4,beta)
    return spdiag([y,u])

def getc(n,dic):
    'generate the linear coefficient.'
    z = getz(n,dic)
    zeros = spmatrix([],[],[],(n*4,1))
    return sparse([-z,zeros])

def get_ydictid(n):
    'Map the position (i,j) of y to the column number of the expanded matrix.'
    l = []
    corner = [(-1,-1),(-1,n),(n,-1),(n,n)]
    for i in range(-1,n+1):
        for j in range(-1,n+1):
            if (i,j) not in corner:
                l.append((i,j))  

    # Create the dictionary
    dictid = dict()
    for val, key in enumerate(l):
        dictid[key] = val
    return dictid

def getz(n,dic):
    'Generate the vector of z'
    h = 1.0/(n-1)
    z = matrix(0.0,(n**2+4*n,1))
    corner = [(-1,-1),(-1,n),(n,-1),(n,n)]
    for i in range(-1,n+1):
        for j in range(-1,n+1):
            if (i,j) not in corner:
                z[dic[(i,j)]] = sin(5*i*h) + cos(4*j*h)

    return z

def identity(n,val = 1):
    'Generate an nxn identity matrix.'
    return spdiag([val]*n)

def getQP(n=10,beta=1.0e-5,phi=0):
    'Generate the QP data structure'
    prob = dict()
    dic = get_ydictid(n)
    prob['H'] = getH(n,beta)
    prob['c'] = getc(n,dic)
    prob['Aeq'], prob['beq'] = geteq(n,dic)
    prob['l'], prob['u'] = getlu(n,phi)
    prob['A'] = sp_rand(0,n**2 + 8*n,0)
    prob['bl'] = sp_rand(0,1,0)
    prob['bu'] = sp_rand(0,1,0)
    prob['x0'] = sp_rand(n**2 + 8*n,1,0.5)
    return QP(prob)

def test_control(n=10,beta=1.0e-5,phi=0):
    'Test function to generate a discretized QP.'
    qp = getQP(n,beta,phi)
    pdas = PDAS(qp,OptTol=1.0e-6)
    pdas.inv_norm = 1.0e+4
    pdas2 = copy(pdas)
    print 'Solving optimal boundary control problem by exact subproblem solve.'
    pdas.exact_solve()
    print 'Solving optimal boundary control problem by inexact subproblem solve.'
    pdas2.inexact_solve()

if __name__ == '__main__':
    n = 10
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])
        print n
    test_control(n)
