from cvxopt import matrix, spmatrix
from randutil import sprandsym, sp_rand
import numpy as np

class QP(object):
    '''
    Class to hold a quadratic optimization problem(QP) of the form:
    min {1/2x'Hx + c'x| bl<=Ax<=bu, Aeq x=beq, l<=x<=u}.
    '''
    def __init__(self,prob={}):
        'Initialize a QP from a dictionary.'
        self.H   = prob.get('H')
        self.c   = prob.get('c')
        self.A   = prob.get('A')
        self.bl  = prob.get('bl')
        self.bu  = prob.get('bu')
        self.Aeq = prob.get('Aeq')
        self.beq = prob.get('beq')
        self.l   = prob.get('l')
        self.u   = prob.get('u')   
        self.x0  = prob.get('x0')
        self._preprocessing()
        self._guess()

    def _preprocessing(self):
        'Preprocess and validate the data, optional'
        self._fill()
        if self.c and self.Aeq and self.A:        
            assert (self.numvar,self.numvar) == self.H.size
            assert (self.numineq,self.numvar) == self.A.size
            assert (self.numeq, self.numvar) == self.Aeq.size
            assert (self.numvar,1) == self.c.size
            assert (self.numineq,1) == self.bl.size
            assert (self.numineq,1) == self.bu.size
            assert (self.numeq,1) == self.beq.size
            assert (self.numvar,1) == self.l.size
            assert (self.numvar,1) == self.u.size
            assert (self.numvar,1) == self.x0.size

    def _fill(self):
        'Obtain the necessary information, initialize their values'
        self.numvar = 0
        self.numeq = 0
        self.numineq = 0

        if self.c:
            self.numvar = self.c.size[0]
        if self.Aeq:
            self.numeq = self.Aeq.size[0]
        if self.A:
            self.numineq = self.A.size[0]

    def _guess(self):
        'Guess an initial partition, either from the initial point or randomness'
        pass

    def __str__(self):
        'Display info of QP when print is called'
        print('This QP has {0} variables, {1} EQs, {2} general InEQs.\n'.format(self.numvar,self.numeq,self.numineq))
        if self.numvar <= 10:
            print('Hessian H:\n')
            print(self.H)
            print('Linear coefficients c:\n')
            print(self.c)
            print('Equality constraints LHS Aeq:\n')
            print(self.Aeq)
            print('Equality constraints RHS beq:\n')
            print(self.beq)
            print('Ineq constraints matrix A:\n')
            print(self.A)
            print('Ineq constraints lower bound bl:\n')
            print(self.bl)
            print('Ineq constraints upper bound bu:\n')
            print(self.bu)
            print('Variational lower bound l:\n')
            print(self.l)
            print('Variational upper bound u:\n')
            print(self.u)
        return ''
            
def randQP(size = 10):
    'Function to test QP class'
    prob = dict()
    n = size
    m = 1
    prob['H']    = sprandsym(n)
    prob['c']    = sp_rand(n,1,1)
    prob['A']    = sp_rand(m,n,1)
    prob['bl']   = sp_rand(m,1,1)
    prob['bu']   = prob['bl'] + 1
    prob['Aeq']  = sp_rand(m,n,0.8)
    prob['beq']  = sp_rand(m,1,1)
    prob['l']    = sp_rand(n,1,1)

    prob['u']    = prob['l'] + 1
    prob['l'][0] = -np.inf
    prob['u'][2] = np.inf
    prob['x0']   = sp_rand(n,1,0.8)
    qp = QP(prob)
    return qp

if __name__ == '__main__':
    'When called from command line.'
    _test_qp()
