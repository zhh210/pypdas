'''
Test different implementation of linear equation solver MINRES
Including:
- a revised version of http://pages.cs.wisc.edu/~kline/cvxopt/ex.minres_py
- scipy.sparse.linalg.minres
'''
from cvxopt import solvers, matrix, mul, normal, setseed
from cvxopt.lapack import *
from cvxopt.blas import *
from pdas.minres import MINRES
import time as t
import cvxopt.misc as misc
import numpy as np
from scipy.sparse.linalg import minres
from pdas.randutil import sp_rand, sprandsym
from cvxopt import sparse, spmatrix
from pdas.convert import cvxopt_to_numpy_matrix, numpy_to_cvxopt_matrix

def test_minres():

    setseed(2)
    n=35
    G=matrix(np.eye(n), tc='d')
    for jj in range(5):
        gg=normal(n,1)
        hh=gg*gg.T
        G+=(hh+hh.T)*0.5
        G+=normal(n,1)*normal(1,n)

    G=(G+G.T)/2

    b=normal(n,1)


    svx=+b
    gesv(G,svx)
    
    tol=1e-10
    show=False
    maxit=None
    t1=t.time()

    # Create a MINRES class
    m = MINRES(G,b)

    m.option['show'] = show
    m.option['rtol'] = tol

    m.solve()

    mg=max(G-G.T)
    if mg>1e-14:sym='No'
    else: sym='Yes'
    alg='MINRES'

    print alg
    print "Is linear operator symmetric? (Symmetry is required) " + sym
    print "n: %3g  iterations:   %3g" % (n, m.iter)
    print " norms computed in ", alg
    print " ||x||  %9.4e  ||r|| %9.4e " %( nrm2(m.x), nrm2(G*m.x -m.b))

def test_minres_scipy():
    H = sprandsym(10)
    v = sp_rand(10,3,0.8)
    A = sparse([[H],[v]])
    vrow = sparse([[v.T],[spmatrix([],[],[],(3,3))]])
    A = sparse([A,vrow])
    b = sp_rand(13,1,0.8)
    As = cvxopt_to_numpy_matrix(A)
    bs = cvxopt_to_numpy_matrix(matrix(b))
    result = minres(As,bs,)
    x = numpy_to_cvxopt_matrix(result[0])
    print nrm2(A*x-b)


if __name__=='__main__':
    test_minres()
    test_minres_scipy()
