'''
Classes controling the behavior of different observers (printer, condioner, monitor, etc.).
'''
from numpy.linalg import norm
from numpy import inf
from time import clock
from cvxopt import matrix
from cvxopt.blas import nrm2, iamax
from scipy.sparse.linalg import minres
from utility import CG

class observer(object):
    '''Base class for observer '''
    def __init__(self,PDAS):
        self.solver = PDAS

    def __call__(self):
        pass

class printer(observer):
    '''Printer class'''
    def __init__(self,PDAS):
        'Attach printer to solver PDAS'
        self.solver = PDAS

    def __call__(self,what = {}):
        'Print out current iteration when called by printer()'
        p = self.solver
        v = self.solver.violations
        if p.iter == 0:
            num = 80
            print '='*num
            print 'Iter     Obj           KKT        |AL|  |AU|  |I|   |V|   |cV|   invnorm  Krylov'
            print '='*num
        print '{iter:>4d}  {obj:^12.2e}  {kkt:^12.2e}  {AL:>4d}  {AU:>4d}  {I:>4d}  {V:>4d}  {cV:>4d}   {inv:>6.2e}  {CGiter:>4d}'.format(iter=p.iter,obj=p._compute_obj(), kkt=p.kkt,AL=len(p.AL),AU=len(p.AU),I=len(p.I),V=p.lenV,cV=p.correctV.lenV,inv=p.inv_norm,CGiter=p.cgiter)

class conditioner(object):
    'Conditioner class'
    option = {
        'CG_res_absolute' : 1.0e-12,
        'CG_res_relative' : 1.0e-3,
        'identified_estimated_ratio' : 0.5,
        'CG_res_absolute_hard': 1.0e-2,
        }
    def __init__(self,iPDAS):
        'Attach conditioner to solver iPDAS'
        self.solver = iPDAS
        self.option = dict(conditioner.option)
#        self.option.update(kwarg)

    def __call__(self,what = {}):
        'Return whether the condition is satisfied'
        satisfied = (
            (norm(self.solver.CG_r,inf) < self.option['CG_res_absolute_hard']
             and (not self.enforce_exact())
             and self.at_least_one()
#             and (self.is_CG_res_absolute() 
#             and self.is_CG_res_relative()
             and self.is_identified_estimate())
#             or self.all_identified()
             or self.is_CG_res_absolute()
        )

        if satisfied:
            print (not self.enforce_exact(),
                   self.is_CG_res_absolute() ,
                   # self.is_CG_res_relative(),
                   self.at_least_one(),
                   self.is_identified_estimate(),
                   # self.all_identified()
                   )

        return satisfied

    def is_CG_res_absolute(self):
        'Linear solver has satisfied absolute tolerance'
        # print 'from observer', max(abs(self.solver.CG_r))
        # print 'from observer', self.solver.state
        # print 'from observer', self.solver.iter

        val = norm(self.solver.CG_r,inf) < self.option['CG_res_absolute']
        if val:
            # Replace violation set 
            self.solver.correctV = self.solver.violations
        return val
    
    def is_CG_res_relative(self):
        'Linear solver has satisfied relative tolerance'
        return norm(self.solver.CG_r,inf)/norm(self.solver.CG_r0,inf) < self.option['CG_res_relative']

    def is_identified_estimate(self):
        '|identified V|/|estimated V| sufficiently large'
        lenIdentified = 0
        lenEstimated = 0
        for key,val in self.solver.correctV.__dict__.iteritems():
            lenIdentified += len(val)
            lenEstimated += len(getattr(self.solver.violations,key))
        #print lenIdentified, lenEstimated
        return lenIdentified >= self.option['identified_estimated_ratio']*lenEstimated

    def at_least_one(self):
        'At least one violation is identified.'
        return self.solver.correctV.lenV >= 1

    def all_identified(self):
        'All violations have been identified'
        default = True
        if self.solver.correctV.lenV == 0:
            default = False
        for key,val in self.solver.correctV.__dict__.iteritems():
            if val != getattr(self.solver.violations,key):
                default = False

        return default

    def enforce_exact(self):
        '''
        Decide if current partition is already optimal, 
        if so, enforce exact solution
        '''
        c = self.solver.correctV
        e = self.solver.violations
        len_c = len(c.Vxl+c.Vxu+c.Vzl+c.Vzu)
        len_e = len(e.Vxl+e.Vxu+e.Vzl+e.Vzu)
        
        val = self.solver.iter != 0 and len_c == 0 and len_e == 0 and self.solver.kkt > self.solver.option['OptTol']
        return val

def estimate_inv_norm(Lhs,rhs = None,x0 = None):
    'Auxilliary function to estimate upper bound of inv norm of Lhs'
    if rhs is None:
        rhs = matrix(1,(Lhs.size[0],1))
    cg = CG(Lhs,rhs,x0)
    Ax = Lhs*cg.x
    v = [i for i in Ax if i < 0]
    while len(v) > 0:# or min(Ax)<1e-16:
        cg.iterate()
        Ax = Lhs*cg.x
        #print matrix([cg.x.T, Ax.T])
        #print min(Ax)
        v = [i for i in Ax if i < 0]
    return (max(abs(cg.x))/min(Ax),cg.iter)

class monitor(observer):
    '''Monitor class'''
    def __init__(self,PDAS,recent=5, est_fun = None):
        self.solver = PDAS
        self.est_fun = est_fun
        self.cgiter = 0
        if est_fun is None:
            self.n = recent
            self.kktlist = [None]*recent

    def __call__(self,what = {}):
        if not self.est_fun is None:
            Lhs,rhs,x0 = self.solver._get_lineq()
            self.solver.inv_norm, cgiter = self.est_fun(Lhs)
            self.cgiter += cgiter
            return
        # Update the kktlist
        pos = self.solver.iter%self.n

        self.kktlist[pos] = self.solver.kkt
        if not (None in self.kktlist):
            # Decide if ref-kkt decreases strictly 
            # cur = max([i%self.n for i in range(pos,pos-self.n+1,-1)])
            cur = self.solver.kkt
            pre = max([self.kktlist[i%self.n] for i in range(pos-1,pos-self.n,-1)])

            if cur >= pre:
                self.solver.inv_norm *= 1.2
                self.kktlist[pos] = pre


class collector(observer):
    '''Collector class'''
    def __init__(self,PDAS):
        self.solver = PDAS
        self.res_relative = 0
        self.res_abs = 0
        self.num = 0
        self.t = clock()

    @property
    def time_elapse(self):
        'Compute the time from initialization until now'
        return clock() - self.t

    def __call__(self,what={}):
        'Collect info of res_raio'
        r = self.solver.CG_r
        r0 = self.solver.CG_r0
        self.res_relative += abs(r[iamax(r)]/r0[iamax(r0)])
        self.res_abs += abs(r[iamax(r)])
        self.num += 1

