'Test file for pdas.toolbox functions.'
from pdas.toolbox import pdas, ipdas
from pdas.prob import randQP

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

if __name__=='__main__':
    'Run from command line'
    test_pdas()
    test_ipdas()
