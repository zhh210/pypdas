'Example to call pdas toolbox functions.'
from pdas.toolbox import pdas,ipdas
from pdas.randutil import sprandsym, sp_rand

def test_eg_pdas():
    'Example to call pdas, with exact ssm solve.'
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
    print('Solving a random QP with exact ssm solve:\n')
    pdas(H,c,Aeq,beq,A,bl,bu,l,u,x0)

    # Solve a bound constrained QP
    print('Solving a random bound-constrained QP with exact ssm solve:\n')
    pdas(H=H,c=c,l=l,u=u,x0=x0)

def test_eg_ipdas():
    'Example to call pdas, with inexact ssm solve.'
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
    print('Solving a random QP with inexact ssm solve:\n')
    ipdas(H,c,Aeq,beq,A,bl,bu,l,u,x0)

    # Solve a bound constrained QP
    print('Solving a random QP with inexact ssm solve:\n')
    ipdas(H=H,c=c,l=l,u=u,x0=x0)


if __name__=='__main__':
    'Run from command line'
    test_eg_pdas()
    test_eg_ipdas()
