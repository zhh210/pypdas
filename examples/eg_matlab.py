'Example to call pdas toolbox functions by reading data from MATLAB.'
from pdas.toolbox import pdas,ipdas
from pdas.randutil import sprandsym, sp_rand
from pdas.matfile import read
from cvxopt import matrix

def test_sparse():
    'Example to call pdas, with sparse Hessian.'

    # Read data from MATLAB file
    data = read('sparse.mat')

    H    = data['H'] 
    c    = data['c'] 
    A    = data['A'] 
    bl   = data['bl'] 
    bu   = data['bu'] 
    Aeq  = data['Aeq'] 
    beq  = data['beq'] 
    l    = data['l'] 
    u    = data['u'] 
    x0   = data['x0'] 

    # Solve a general QP
    print('Sparse H: Solving a random QP with exact ssm solve:\n')
    pdas(H,c,Aeq,beq,A,bl,bu,l,u,x0)

    # Solve a bound constrained QP
    print('Sparse H: Solving a random bound-constrained QP with exact ssm solve:\n')
    pdas(H=H,c=c,l=l,u=u,x0=x0)

    # Solve a general QP
    print('Sparse H: Solving a random QP with inexact ssm solve:\n')
    ipdas(H,c,Aeq,beq,A,bl,bu,l,u,x0)

    # Solve a bound constrained QP
    print('Sparse H: Solving a random bound-constrained QP with inexact ssm solve:\n')
    ipdas(H=H,c=c,l=l,u=u,x0=x0)

def test_dense():
    'Example to call pdas, with sparse Hessian.'

    # Read data from MATLAB file
    data = read('sparse.mat')

    H    = matrix(data['H'])
    c    = data['c'] 
    A    = data['A'] 
    bl   = data['bl'] 
    bu   = data['bu'] 
    Aeq  = data['Aeq'] 
    beq  = data['beq'] 
    l    = data['l'] 
    u    = data['u'] 
    x0   = data['x0'] 

    # Solve a general QP
    print('Dense H: Solving a random QP with exact ssm solve:\n')
    pdas(H,c,Aeq,beq,A,bl,bu,l,u,x0)

    # Solve a bound constrained QP
    print('Dense H: Solving a random bound-constrained QP with exact ssm solve:\n')
    pdas(H=H,c=c,l=l,u=u,x0=x0)

    # Solve a general QP
    print('Dense H: Solving a random QP with inexact ssm solve:\n')
    ipdas(H,c,Aeq,beq,A,bl,bu,l,u,x0)

    # Solve a bound constrained QP
    print('Dense H: Solving a random bound-constrained QP with inexact ssm solve:\n')
    ipdas(H=H,c=c,l=l,u=u,x0=x0)


if __name__=='__main__':
    'Run from command line'
    test_sparse()
    test_dense()
