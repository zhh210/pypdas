'''
Test of call_back() using function object for scipy.sparse.linalg.minres.
'''
from scipy.sparse.linalg import minres
from cvxopt import matrix
from pdas.pdas import PDAS
from pdas.prob import randQP
from pdas.convert import numpy_to_cvxopt_matrix, cvxopt_to_numpy_matrix
import ctypes
import inspect

class LS(object):
    'Class holder for linear equation object.'
    def __init__(self):
        qp = randQP(10)
        self.pdas = PDAS(qp)
        self.Lhs, self.rhs, self.x0 = self.pdas._get_lineq()
        self.it = 0

    def __call__(self,x):
        # Access value of itn in minres
        frame = inspect.currentframe().f_back
        self.it = frame.f_locals['itn']
        # Change value of
        # ctype.pythonapi.Pycell_Set(id(inner.func_closure[0]), id(x))
        print('solution updated to: ', self.it)
        

if __name__=='__main__':
    a = LS()
    Lhs = cvxopt_to_numpy_matrix(a.Lhs)
    rhs = cvxopt_to_numpy_matrix(matrix(a.rhs))
    x0  = cvxopt_to_numpy_matrix(matrix(a.x0))
    minres(Lhs,rhs,x0,callback = a)


