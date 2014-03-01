'Projected CG method to solve equality-constrained QPs.'
from cvxopt import matrix, spmatrix, cholmod, sparse, solvers
from randutil import sprandsym, sp_rand
from cvxopt.blas import dotu, nrm2
from cvxopt.umfpack import linsolve
from copy import copy
from pdb import set_trace
def pcg0(H,c,A,b,x0,fA=None,callback=None):
    '''
    Projected CG method to solve the problem: {min 1/2x'Hx + c'x | Ax = b}.
    Initial point x0 must safisty Ax0 = b. Unstable version, not recommended.
    '''
    # Initialize some variables
    r = H*x0 + c
    r = project(A,r)
    g = project(A,r)
    p = -copy(g)
    x = copy(x0)

    while True:
        alpha = dotu(r,g)/dotu(p,H*p)
        x = x+ alpha*p
        r2 = r + alpha*H*p
        g2 = project(A,r2)
        # Do iterative refinement
        # for i in range(5000):
        #     g2 = project(A,g2)
        beta = dotu(r2,g2)/dotu(r,g)
        p = -g2 + beta*p
        g = copy(g2)
        r = copy(r2)
        if nrm2(r) < 1e-16:
            break
    return x

def pcg(H,c,A,b,x0,fA=None,callback=None):
    '''
    Projected CG method to solve the problem: {min 1/2x'Hx + c'x | Ax = b}.
    Initial point x0 must safisty Ax0 = b. Stable version, recommended.
    '''
    # Initialize some variables
    r = H*x0 + c
    y = minG(A,r)
    r = r - A.T*y
    g = project(A,r)
    p = -copy(g)
    x = copy(x0)

    while True:
        alpha = dotu(r,g)/dotu(p,H*p)

        x = x+ alpha*p
        r2 = r + alpha*H*p
        y = minG(A,r2)
        r2 = r2 - A.T*y
        g2 = project(A,r2)
        # Do iterative refinement
        # for i in range(50):
        #     g2 = project(A,g2)
        beta = dotu(r2,g2)/dotu(r,g)
        p = -g2 + beta*p
        g = copy(g2)
        r = copy(r2)
        if abs(dotu(r,g)) < 1e-12:
            break
    return x
        
def project(A,r,G = None, fA = None):
    'Project r to null(A) by solving the normal equation AA.t v = Ar.'
    m,n = A.size
    if G is None:
        G = spmatrix([1]*n,range(n),range(n))
    Lhs1 = sparse([G,A])
    Lhs2 = sparse([A.T, spmatrix([],[],[],(m,m))])
    Lhs = sparse([[Lhs1],[Lhs2]])
    rhs = matrix([r,spmatrix([],[],[],(m,1))])
    linsolve(Lhs,rhs)
    return rhs[:n]
    
def minG(A,r,G= None):
    'Find y that minimize the norm(r - A.T*y)'
    if G is None:
        Lhs = A*A.T
        rhs = A*r
        cp = copy(rhs)
        linsolve(Lhs,rhs)
        return rhs

def test_pcg():
    'Test function for projected CG.'
    n = 10
    m = 4
    H = sprandsym(n,n)
    A = sp_rand(m,n,0.9)
    x0 = matrix(1,(n,1))
    b = A*x0
    c = matrix(1.0,(n,1))

    x_pcg = pcg(H,c,A,b,x0)


    Lhs1 = sparse([H,A])
    Lhs2 = sparse([A.T,spmatrix([],[],[],(m,m))])
    Lhs = sparse([[Lhs1],[Lhs2]])
    rhs = -matrix([c,spmatrix([],[],[],(m,1))])
    rhs2 = copy(rhs)
    linsolve(Lhs,rhs)
    #print rhs[:10]


    sol = solvers.qp(H,c,A=A,b=b)
    print ' cvxopt qp|   pCG'
    print matrix([[sol['x']],[x_pcg]])
    print 'Dual variables:'
    print sol['y']
    print 'KKT equation residuals:'
    print H*sol['x'] + c + A.T*sol['y']

if __name__=='__main__':
    'Call from cmd line'
    test_pcg()
