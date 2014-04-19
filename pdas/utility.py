'''
Utility classes and functions that are extensively used in PDAS.
'''
import re, pdb
import cvxopt
import inspect, ctypes
import numpy as np
from cvxopt import matrix, spmatrix, spdiag, lapack
from randutil import sprandsym, sp_rand
from copy import copy
from numpy import inf
from scipy.sparse.linalg import minres
from convert import numpy_to_cvxopt_matrix, cvxopt_to_numpy_matrix
from math import sqrt
from cvxopt.blas import nrm2, dot, asum
import observer as obs

locals_to_fast = ctypes.pythonapi.PyFrame_LocalsToFast
locals_to_fast.restype = None
locals_to_fast.argtypes = [ctypes.py_object, ctypes.c_int]


class optimset(object):
    '''
    Info : Class for holding default parameters and setting values.
    Msg  : Default parameters:
           - 'Solver': 'CG',
           - 'OptTol': 1.0e-7,
           - 'ResTol': 1.0e-10,
           - 'MaxItr': 1000,
           - 'eps'   : 1.0e-16
    '''
    default_options = {
        'Solver': 'CG',
        'OptTol': 1.0e-6,
        'ResTol': 1.0e-12,
        'MaxItr': 1000,
        'fun_estinv': None,
        'eps': 1.0e-16,
        }

    def __init__(self, **kwargs):
        self.options = dict(optimset.default_options)
        self.options.update(kwargs)

    def __getitem__(self,key):
        return self.options[key]


class Violations(object):
    'Class to hold a all violation indices'
    def __init__(self,):
        self.Vxl  = []
        self.Vxu  = []
        self.Vxcl = []
        self.Vxcu = []
        self.Vzl  = []
        self.Vzu  = []
        self.Vzcl = []
        self.Vzcu = []

    def _get_lenV(self):
        return len(self.Vxl + self.Vxu+self.Vxcl+self.Vxcu+self.Vzl+self.Vzu+self.Vzcl+self.Vzcu)

    lenV = property(_get_lenV)

class CG(object):
    'Apply conjugate-gradient method to solve Ax=b'
    def __init__(self,A,b,x0 = None):
        self.A = A
        self.b = b
        self.n = A.size[0]
        self.iter = 0
        if x0 is not None:
            self.x = x0
        else:
            self.x = matrix(1.0,b.size)

        self.r0 = A*self.x - b
        self.r = copy(self.r0)
        self.p = copy(-self.r)
        self.inv_norm = sum(self.x**2)/sum((self.b + self.r)**2)
        
    def iterate(self,times = 1):
        'Apply cg iterations on the linear system'
        for i in range(times):
            Ap = self.A*self.p
            alpha = sum(self.r**2)/(self.p.T*Ap)
            alpha = alpha[0]

            self.x = self.x + alpha*self.p
            self.rpre = copy(self.r)
            self.r = self.r + alpha*Ap
            beta = sum(self.r**2)/sum(self.rpre**2)

            self.p =  -self.r + beta*self.p
            self.iter += 1
            # print max(abs(self.r))

            # Use this iteration to generate an estimate of norm(invA)
            self.inv_norm = max(sum(self.x**2)/sum((self.b + self.r)**2),self.inv_norm)
        return self.x

    def solve(self,cond):
        'Apply CG iterations until condition cond satisfied'
        while not cond():
            self.iterate()

        return self.x

    def cond_res(self,restol=1.0e-10):
        'Condition on whether residual is sufficiently small'
        return max(abs(self.r)) < restol

class TetheredLinsys(object):
    '''
    A composation of PDAS, and linear solver.
    Call linear solver and apply changes on PDAS.xs

    '''
    def __init__(self,PDAS,linsol):
        'Attach PDAS, and linear equation solver '
        self.linsol = linsol
        self.PDAS = PDAS
        self.PDAS.CG_r0 = linsol.r0
        self.PDAS.CG_r = linsol.r0
        self.PDAS.correctV = Violations()

    def iter(self,n = 1):
        'Make linsol apply n iterations, apply changes on solver'
        self._linsol(n)
        self.PDAS.CG_r = self.linsol.r
#        print 'from utility', max(abs(self.linsol.r))

        if self.PDAS.inv_norm is None:
            self.PDAS.inv_norm = 1.1*self.linsol.inv_norm
        else:
            self.PDAS.inv_norm = max(self.PDAS.inv_norm, 1.1*self.linsol.inv_norm)
        self._generate_bounds()
        return (self.solution_dist_lb,self.solution_dist_ub)

    def _linsol(self,times = 1):
        # Obtain inexact solution by applying times iterations
        xy = self.linsol.iterate(times)
        nI = len(self.PDAS.I)
        ny = self.PDAS.QP.numeq
        ncL = len(self.PDAS.cAL)
        ncU = len(self.PDAS.cAU)
        self.PDAS.x[self.PDAS.I] = xy[0:nI]
        self.PDAS.y = xy[nI:nI + ny]
        if self.PDAS.czl.size[0] > 0:
            self.PDAS.czl[self.PDAS.cAL] = xy[nI+ny:nI+ny+ncL]
            self.PDAS.czu[self.PDAS.cAU] = xy[nI+ny+ncL:]

    def _generate_bounds(self, B = None):
        'A general obtain bounds from an estimate of norm(invHii): B'
        if B is None or self.PDAS.inv_norm is None:
            B = 1.1*self.linsol.inv_norm

        viration = nrm2(self.linsol.r)*B*matrix(1.0,self.linsol.r0.size)

        self.solution_dist_lb = - viration
        self.solution_dist_ub = viration


class LinsysWrap(object):
    'A wrapper of liear equation solver to terminate when exactness is obtained'
    def __init__(self,PDAS,lin_solver = minres):
        self.lin_solver = minres
        self.PDAS = PDAS
        self.Lhs, self.rhs, self.x0 = PDAS._get_lineq()
        self.r0 = self.Lhs*self.x0 - self.rhs
        self.PDAS.CG_r0 = copy(self.r0)
        self.PDAS.CG_r = copy(self.r0)
        self.PDAS.correctV = Violations()
        self.err_lb = None
        self.err_ub = None
        self.inv_norm = 1e+2
        self.iter = 0

    def solve(self):
        'Solve the equation until certain conditions are satisfied'
        A = cvxopt_to_numpy_matrix(self.Lhs)
        b = cvxopt_to_numpy_matrix(matrix(self.rhs))
        x0 = cvxopt_to_numpy_matrix(self.x0)
        self.lin_solver(A,b,x0,tol=1.0e-16,maxiter=1e+9,callback=self.callback)

    def callback(self,xk = None):
        'Callback function after each iteration of minres'

        # Access current iteration from lin_solver
        
        xy = numpy_to_cvxopt_matrix(xk)

        # Set x[I], y, and czl czl(if nonempty)
        nI = len(self.PDAS.I)
        ny = self.PDAS.QP.numeq
        ncL = len(self.PDAS.cAL)
        ncU = len(self.PDAS.cAU)
        self.PDAS.x[self.PDAS.I] = xy[0:nI]
        self.PDAS.y = xy[nI:nI + ny]
        if self.PDAS.czl.size[0] > 0:
            self.PDAS.czl[self.PDAS.cAL] = xy[nI+ny:nI+ny+ncL]
            self.PDAS.czu[self.PDAS.cAU] = xy[nI+ny+ncL:]

        # Set residuals, and inv_norm estimate
        self.PDAS.CG_r = self.Lhs*xy - self.rhs
        self.inv_norm = max(nrm2(xy)/nrm2(self.rhs + self.PDAS.CG_r),self.inv_norm)

        # Obtain bounds from an estimate of norm(invHii): B
        # if self.PDAS.inv_norm is None or len(self.PDAS._ObserverList['monitor']) < 1:
        #     print 'dynamic'
        #     self.PDAS.inv_norm = max(self.PDAS.inv_norm, 1.1*self.inv_norm)

        B = self.PDAS.inv_norm

        viration = nrm2(self.PDAS.CG_r)*B*matrix(1.0,self.r0.size)

        self.err_lb = - viration
        self.err_ub = viration

        # Update other variables
        self.PDAS._back_substitute()
        self.PDAS.identify_violation_inexact(self.err_lb,self.err_ub)
        frame = inspect.currentframe().f_back
        self.iter = frame.f_locals['itn']


        # If condition satisfied, terminate the linear equation solve
        if self.PDAS.ask('conditioner') is True:
            set_in_frame(inspect.currentframe().f_back,'istop',9)
            # ctypes.pythonapi.PyCell_Set(id(istop),9)

class LinsysWrap_c(LinsysWrap):
    'Special wrapper for optimal control problems'
    def __init__(self,PDAS,lin_solver = minres):
        # Initialize LinsysWrap
        super(LinsysWrap_c,self).__init__(PDAS,lin_solver)
        self.Lhs, self.rhs, self.x0 = PDAS._get_lineq_c()
        # Initialize some other
        pdas = self.PDAS
        qp = pdas.QP
        self.SI = matrix([qp.H[pdas.F,pdas.realI], qp.Aeq[:,pdas.realI]])
        self.SA = matrix([qp.H[pdas.F,pdas.AU], qp.Aeq[:,pdas.AU]])

        # Compute inv(Q)*SI
        QinvSI = copy(self.SI)
        lapack.sytrs(pdas.Q,pdas.ipiv,QinvSI)

        self.RI = qp.H[pdas.realI,pdas.realI] - self.SI.T*QinvSI

    # Override the callback function
    def callback(self,xk = None):
        'Callback function after each iteration of minres'

        # Access current iteration from lin_solver
        
        xy = numpy_to_cvxopt_matrix(xk)

        # Set x[I], y, and czl czl(if nonempty)
        nI = len(self.PDAS.I)
        ny = self.PDAS.QP.numeq
        ncL = len(self.PDAS.cAL)
        ncU = len(self.PDAS.cAU)
        self.PDAS.x[self.PDAS.realI + self.PDAS.F] = xy[:nI]
        self.PDAS.y = xy[nI:]
        if self.PDAS.czl.size[0] > 0:
            self.PDAS.czl[self.PDAS.cAL] = xy[nI+ny:nI+ny+ncL]
            self.PDAS.czu[self.PDAS.cAU] = xy[nI+ny+ncL:]

        # Set residuals, and inv_norm estimate
        self.PDAS.CG_r = self.Lhs*xy - self.rhs

        # Compute matrix inverse norm
        if self.RI.size != (0,0):
#            gamma = obs.estimate_inv_norm(self.RI)[0]
            gamma = 1/mineig(self.RI)[0]
        else:
            gamma = 0

        # Update on pdas
        self.PDAS.inv_norm = gamma

        # Compute tilde v
        Qinvv = self.PDAS.CG_r[len(self.PDAS.realI):]
        rI = self.PDAS.CG_r[:len(self.PDAS.realI)]
        lapack.sytrs(self.PDAS.Q, self.PDAS.ipiv, Qinvv)

        viration = nrm2(rI -self.SI.T*Qinvv)*gamma*matrix(1.0,(len(self.PDAS.realI),1) )

        self.err_lb = - viration
        self.err_ub = viration

        # Update other variables
        self.PDAS._back_substitute()

        # Caveat z_A shifted
        self.PDAS.zu[self.PDAS.AU] = self.PDAS.zu[self.PDAS.AU] + self.SA.T*Qinvv

        self.PDAS.identify_violation_inexact_c(self.err_lb,self.err_ub)
        frame = inspect.currentframe().f_back
        self.iter = frame.f_locals['itn']


        # If condition satisfied, terminate the linear equation solve
        if self.PDAS.ask('conditioner') is True:
            set_in_frame(inspect.currentframe().f_back,'istop',9)
            # ctypes.pythonapi.PyCell_Set(id(istop),9)
    


def set_in_frame(frame, name, value):
    frame.f_locals[name] = value
    locals_to_fast(frame, 1)

def union(x,y):
    'Set operation: return elements either in x or y in ascending order'
    return list(set(x)|set(y))

def intersect(x,y):
    'Set operation: return elements both in x or y in ascending order'
    return list(set(x) & set(y))

def setdiff(x,y):
    'Set operation: return elements in x but not in y in ascending order'
    return [i for i in set(x) if i not in set(y)]

#@profile
def estimate_range(A,xl,xu):
    '''
    Estimate the range of Ax, where xl <= x <= xu
    Mathematically, the lower and upper bounds are:
    '''
    # Decide if A is sparse
    # if 'sp' in str(type(A)) or 'sparse' in str(type(A)):
    #    pass


    # Efficient, exploit the sparsity of A
    Apos = copy(A)
    Aneg = copy(A)

    # For sparse matrices
    Apos.V = cvxopt.max(Apos.V,0)
    Aneg.V = cvxopt.min(Aneg.V,0)
#    pdb.set_trace()
    bl = Apos*xl + Aneg*xu
    bu = Apos*xu + Aneg*xl

    # Relatively efficient, but dense matrix means overhead
    # bl = cvxopt.min(A,0)*xu + cvxopt.max(A,0)*xl
    # bu = cvxopt.min(A,0)*xl + cvxopt.max(A,0)*xu

    # This version is less efficient
    # row, col = A.size
    # bl = matrix(10.0,size=(row,1))
    # bu = matrix(10.0,size=(row,1))
    # for i in range(row):
    #     for j in range(col):
    #         if A[i,j] > 0.0:
    #             bl[i] += A[i,j]*xl[j]
    #             bu[i] += A[i,j]*xu[j]
    #         elif A[i] < 0.0:
    #             print A.size, xl.size, xu.size, i,j
    #             bl[i] += A[i,j]*xu[j]
    #             bu[i] += A[i,j]*xl[j]

    return (bl,bu)

def poweriter(H,x = None,maxit= 1000,tol=1.0e-2):
    'Function to estimate the spectral norm of a matrix.'
    n = H.size[1]
    if x is None:
        x = matrix(1.0,(n,1))/sqrt(n)

    vec = H*x
    norm = dot(x,vec)
    x = vec/nrm2(vec)

    it = 0
    for it in range(maxit):
        norm_pre = copy(norm)
        vec = H*x
        x = vec/nrm2(vec)
        norm = dot(x,vec)
        if abs(norm-norm_pre) < tol:
            break

    return (norm,it,x)

def mineig(H,x = None, maxit = 1000, tol=1.0e-2):
    'Function to use power iteration to obtain mineig of a pd matrix.'

    # Upper bound of the maximum eig
    n = H.size[1]
    nrm1 = 0
    for i in range(n):
        nrm1 = max(nrm1,asum(matrix(H[i,:])))

#    diag = float(np.linalg.norm(H,1))
    diag = nrm1
    Diag = spdiag([diag]*n)
    norm, it,x = poweriter(Diag - H, x, maxit, tol)
    norm = diag - norm
    return (norm,it,x)
    
# Functions for internal test

def _test_optimset():
    'Test function of optimset'
    print 'Testing optimset:'
    print 'Default options:'
    option = optimset()
    for key,item in option.options.iteritems():
        print key,':', item
    print 'Set options: Solver=pdas, NewOpt=1'
    option = optimset(Solver='pdas',NewOpt=1)
    for key in option.options.keys():
        print key,':', option[key]
    print '-'*50


def _test_CG():
    'Test function of CG'

    A = sprandsym(10,10,1)
    b = sp_rand(10,1,0.8)
    cg = CG(A,b)
    cg.solve(cg.cond_res)

def _test_range():
    'Test function of range'
    A = matrix(range(16),(4,4))
    xl = matrix(-1,(4,1))
    xu = matrix(1,(4,1))
    bl, bu = estimate_range(A,xl,xu)
    print bl, bu
    
if __name__ == '__main__':
    _test_optimset()
    _test_CG()
    _test_range()
