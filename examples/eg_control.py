'''
An example demonstrating solving a descritized optimal control problem.
'''
from cvxopt import spmatrix, sparse, spdiag, matrix
from numpy import inf
from math import sin, cos
from copy import copy

from pdas.pdas import PDAS
from prob import QP
from randutil import sp_rand

import pdb

def Laplace(n):
    'create discretized Laplace operator (in 2 dimensions).'
    # Initialize the matrix by a sparse matrix
    L = spmatrix([],[],[],(n**2,n**2))

    # Decide the coefficient matrix
    for i in range(0,n):
        for j in range(0,n):
            # At the inner of the
            y = spmatrix([],[],[],(n,n))
            y[i,j] = -4
            if i != 0:
                y[i-1,j] = 1
            if i != n-1:
                y[i+1,j] = 1
            if j != 0:
                y[i,j-1] = 1
            if j != n-1:
                y[i,j+1] = 1

            y.size = (1,n**2)
            L[getid(i,j,n),:] = y

    return L

def geteq(n):
    'generate the equality constraints.'
    Aeq = Laplace(n)
    h = 1.0/(n+1)
    eq2 = identity(n**2,h**2)
    Aeq = sparse([[Aeq],[eq2]])
    beq = spmatrix([],[],[],(n**2,1))
    return (Aeq, beq)

def getlu(n,phi):
    'generate the variational upper bound.'
    l = matrix(-inf,(2*n**2,1))
    u = matrix([matrix(inf,(n**2,1)), matrix(phi,(n**2,1))])
    return (l,u)

def getH(n,beta):
    'generate the Hessian.'
    y = identity(n**2)
    u = identity(n**2,beta)
    return spdiag([y,u])

def getc(n):
    'generate the linear coefficient.'
    z = getz(n)
    zeros = spmatrix([],[],[],(n**2,1))
    return sparse([-z,zeros])

def getid(i,j,n):
    'map the position (i,j) to the row number of the expanded matrix L.'
    return  j*n + i

def getz(n):
    'Generate the vector of z'
    h = 1.0/(n+1)
    z = matrix(0.0,(n,n))
    for i in range(n):
        for j in range(n):
            z[i,j] = sin(5*(i+1)*h) + cos(4*(j+1)*h)

    z.size = (n**2,1)
    return z

def identity(n,val = 1):
    'Generate an nxn identity matrix.'
    return spdiag([val]*n)

def getQP(n=10,beta=1.0e-5,phi=0):
    'Generate the QP data structure'
    prob = dict()
    prob['H'] = getH(n,beta)
    prob['c'] = getc(n)
    prob['Aeq'], prob['beq'] = geteq(n)
    prob['l'], prob['u'] = getlu(n,phi)
    prob['A'] = sp_rand(0,2*n**2,0)
    prob['bl'] = sp_rand(0,1,0)
    prob['bu'] = sp_rand(0,1,0)
    prob['x0'] = sp_rand(2*n**2,1,0.5)
    return QP(prob)

def test_control(n=10,beta=1.0e-5,phi=0):
    'Test function to generate a discretized QP.'
    qp = getQP(n,beta,phi)
    pdas = PDAS(qp)
    pdas2 = copy(pdas)
    print 'Solving optimal control problem by exact subproblem solve.'
    pdas.exact_solve()
    print 'Solving optimal control problem by inexact subproblem solve.'
    pdas2.inexact_solve()

if __name__ == '__main__':
    test_control()
