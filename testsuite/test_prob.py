from prob import QP
from randutil import sprandsym, sp_rand

def test_qp():
    'Function to test QP class'
    prob = dict()
    n = 5
    m = 1
    prob['H']    = sprandsym(n)
    prob['c']    = sp_rand(n,1,0.8)
    prob['A']    = sp_rand(m,n,0.8)
    prob['bl']   = sp_rand(m,1,1)
    prob['bu']   = prob['bl'] + 1
    prob['Aeq']  = sp_rand(m,n,0.8)
    prob['beq']  = sp_rand(m,1,1)
    prob['l']    = sp_rand(n,1,1)
    prob['u']    = prob['l'] + 1
    prob['x0']   = sp_rand(n,1,0.8)
    qp = QP(prob)
    print(qp)
    return qp

if __name__ == '__main__':
    'When called from command line.'
    test_qp()
