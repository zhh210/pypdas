'''
Interface for PDAS toolbox functions.
'''

from pdas import PDAS
from utility import OptOptions
from prob import randQP, QP
from randutil import sp_rand, sprandsym
from numpy import inf
from cvxopt import matrix

def _generateQP(H,c,Aeq,beq,A,bl,bu,l,u,x0):
    'Auxilliary function for generating QP blass'
    prob = dict()
    prob['H'] = H
    prob['c'] = c
    prob['Aeq'] = Aeq
    prob['beq'] = beq
    prob['A'] = A
    prob['bl'] = bl
    prob['bu'] = bu
    prob['l'] =  l
    prob['u'] = u
    prob['x0'] = x0
    qp = QP(prob)
    return qp

def _default(Aeq,beq,A,bl,bu,l,u,x0,n):
    'Auxilliary function to set default values for some coefficients'
    if Aeq is None:
        Aeq = sp_rand(0,n,0)
    if beq is None:
        beq = sp_rand(0,1,0)
    if A is None:
        A = sp_rand(0,n,0)
    if bl is None:
        bl = sp_rand(0,1,-inf)
    if bu is None:
        bu = sp_rand(0,1,inf)
    if l is None:
        l = matrix(-inf,(n,1))
    if u is None:
        u = matrix(inf,(n,1))
    if x0 is None:
        x0 = sp_rand(n,1,0.8)

    return Aeq,beq,A,bl,bu,l,u,x0


def pdas(H=None, c=None, Aeq=None, beq=None, A=None, bl=None, bu=None, l = None, u = None, x0=None, option = OptOptions()):
    '''
    PDAS algorithm with exact ssm solve.
    Example:
        n = 100  # number of variables
        m = 1    # number of equality
        mi = 1   # number of inequality
        # Generate some random data
        H    = sprandsym(n)
        c    = sp_rand(n,1,1)
        A    = sp_rand(mi,n,1)
        bl   = sp_rand(mi,1,1)
        bu   = bl + 1
        Aeq  = sp_rand(m,n,0.8)
        beq  = sp_rand(m,1,1)
        l    = sp_rand(n,1,1)
        u    = l + 1
        x0   = sp_rand(n,1,0.8)

        # Solve a general QP
        pdas(H,c,Aeq,beq,A,bl,bu,l,u,x0)

        # Solve a bound constrained QP
        pdas(H=H,c=c,l=l,u=u,x0=x0)
    '''
    # Preprocess some coefficients
    n = H.size[0]
    Aeq,beq,A,bl,bu,l,u,x0 = _default(Aeq,beq,A,bl,bu,l,u,x0,n)

    # Create QP blass and solve with exact PDAS
    qp = _generateQP(H,c,Aeq,beq,A,bl,bu,l,u,x0)
    #import pdb
    #pdb.set_trace()
    #print qp.H.size,qp.A.size,qp.Aeq.size,qp.bl.size,qp.bu.size
    pdas = PDAS(qp)
    pdas.exact_solve()

def ipdas(H=None, c=None, Aeq=None, beq=None, A=None, bl=None, bu=None, l = None, u = None, x0=None, option = OptOptions()):
    '''
    PDAS algorithm with inexact ssm solve
    Example:
        n = 100  # number of variables
        m = 1    # number of equality
        mi = 1   # number of inequality
        # Generate some random data
        H    = sprandsym(n)
        c    = sp_rand(n,1,1)
        A    = sp_rand(mi,n,1)
        bl   = sp_rand(mi,1,1)
        bu   = bl + 1
        Aeq  = sp_rand(m,n,0.8)
        beq  = sp_rand(m,1,1)
        l    = sp_rand(n,1,1)
        u    = l + 1
        x0   = sp_rand(n,1,0.8)

        # Solve a general QP
        ipdas(H,c,Aeq,beq,A,bl,bu,l,u,x0)

        # Solve a bound constrained QP
        ipdas(H=H,c=c,l=l,u=u,x0=x0)
    '''
    # Preprocess some coefficients
    n = H.size[0]
    Aeq,beq,A,bl,bu,l,u,x0 = _default(Aeq,beq,A,bl,bu,l,u,x0,n)

    # Create QP blass and solve with inexact PDAS
    qp = _generateQP(H,c,Aeq,beq,A,bl,bu,l,u,x0)
    pdas = PDAS(qp)
    pdas.inexact_solve()

def test_pdas():
    'Test function for pdas()'
    # Generate a random QP
    qp = randQP(size = 100, numeq = 1, numineq = 1)
    pdas(qp.H,qp.c,qp.Aeq,qp.beq,qp.A,qp.bl,qp.bu,qp.l,qp.u,qp.x0)

    # Solve the random bound constrained QP
    pdas(qp.H,qp.c,l=qp.l,u=qp.u,x0=qp.x0)

def test_ipdas():
    '''
    Test function for ipdas()
    '''

    # Generate a random QP
    qp = randQP(size = 100, numeq = 1, numineq = 1)
    ipdas(qp.H,qp.c,qp.Aeq,qp.beq,qp.A,qp.bl,qp.bu,qp.l,qp.u,qp.x0)

    # Solve the random bound constrained QP
    ipdas(qp.H,qp.c,l=qp.l,u=qp.u,x0=qp.x0)

def test_eg_pdas():
    'Test function for the instruction'
    if True:
        n = 100  # number of variables
        m = 1    # number of equality
        mi = 1   # number of inequality
        # Generate some random data
        H    = sprandsym(n)
        c    = sp_rand(n,1,1)
        A    = sp_rand(mi,n,1)
        bl   = sp_rand(mi,1,1)
        bu   = bl + 1
        Aeq  = sp_rand(m,n,0.8)
        beq  = sp_rand(m,1,1)
        l    = sp_rand(n,1,1)
        u    = l + 1
        x0   = sp_rand(n,1,0.8)

        # Solve a general QP
        pdas(H,c,Aeq,beq,A,bl,bu,l,u,x0)

        # Solve a bound constrained QP
        pdas(H=H,c=c,l=l,u=u,x0=x0)

def test_eg_ipdas():
    'Test function for the instruction'
    if True:
        n = 100  # number of variables
        m = 1    # number of equality
        mi = 1   # number of inequality
        # Generate some random data
        H    = sprandsym(n)
        c    = sp_rand(n,1,1)
        A    = sp_rand(mi,n,1)
        bl   = sp_rand(mi,1,1)
        bu   = bl + 1
        Aeq  = sp_rand(m,n,0.8)
        beq  = sp_rand(m,1,1)
        l    = sp_rand(n,1,1)
        u    = l + 1
        x0   = sp_rand(n,1,0.8)

        # Solve a general QP
        ipdas(H,c,Aeq,beq,A,bl,bu,l,u,x0)

        # Solve a bound constrained QP
        ipdas(H=H,c=c,l=l,u=u,x0=x0)


if __name__=='__main__':
    'Run from command line'
    test_pdas()
    test_ipdas()
    test_eg_pdas()
    test_eg_ipdas()
