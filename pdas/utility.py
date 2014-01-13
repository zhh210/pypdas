'''
Utility classes and functions that are extensively used in PDAS.
'''
import re
import cvxopt
from cvxopt import matrix, spmatrix
from randutil import sprandsym, sp_rand
from copy import copy
from numpy import inf



class OptOptions(object):
    '''
    Info : Class for holding default parameters and setting values.
    Msg  : Default parameters:
           - 'Solver': 'CG',
           - 'OptTol': 1.0e-10,
           - 'ResTol': 1.0e-8,
           - 'MaxItr': 1000,
    '''
    default_options = {
        'Solver': 'CG',
        'OptTol': 1.0e-10,
        'ResTol': 1.0e-8,
        'MaxItr': 1000,
        }

    def __init__(self, **kwargs):
        self.options = dict(OptOptions.default_options)
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

        viration = sum(self.linsol.r**2)*B*matrix(1.0,self.linsol.r0.size)

        self.solution_dist_lb = - viration
        self.solution_dist_ub = viration



def union(x,y):
    'Set operation: return elements either in x or y in ascending order'
    return list(set(x)|set(y))

def intersect(x,y):
    'Set operation: return elements both in x or y in ascending order'
    return list(set(x) & set(y))

def setdiff(x,y):
    'Set operation: return elements in x but not in y in ascending order'
    return [i for i in set(x) if i not in set(y)]

def estimate_range(A,xl,xu):
    'Estimate the range of Ax, where xl <= x <= xu'
    bl = cvxopt.min(A,0)*xu + cvxopt.max(A,0)*xl
    bu = cvxopt.min(A,0)*xl + cvxopt.max(A,0)*xu
    return (bl,bu)

# Functions for internal test

def _test_OptOptions():
    'Test function of OptOptions'
    print 'Testing OptOptions:'
    print 'Default options:'
    option = OptOptions()
    for key,item in option.options.iteritems():
        print key,':', item
    print 'Set options: Solver=pdas, NewOpt=1'
    option = OptOptions(Solver='pdas',NewOpt=1)
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
    _test_OptOptions()
    _test_CG()
    _test_range()
