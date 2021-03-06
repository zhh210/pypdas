'''
Interface of primal-dual active-set method.
'''
import pdb,os, random, string, time
import observer as obs
from prob import QP, randQP
from solver import solver
from scipy.sparse.linalg import minres
import numpy as np
import utility as ut
from utility import setdiff, intersect, union, CG, estimate_range, TetheredLinsys, Violations, LinsysWrap, LinsysWrap_c
from cvxopt import matrix, spmatrix, sparse, cholmod, lapack, spdiag
from copy import copy
from convert import cvxopt_to_numpy_matrix, numpy_to_cvxopt_matrix
import inspect, ctypes
from matfile import read, write
#from pymatbridge import Matlab

locals_to_fast = ctypes.pythonapi.PyFrame_LocalsToFast
locals_to_fast.restype = None
locals_to_fast.argtypes = [ctypes.py_object, ctypes.c_int]

class PDAS(object):

    # Defaultly observers
    observerlist = {
        'printer'     : [],
        'collector'   : [],
        'conditioner' : [],
        'monitor'     : [],
        'modifier'    : []
        }

    def __init__(self,QP = QP(),**kwargs):
        #super(solver,self).__init__(self,QP,**kwargs)
        self.QP = QP
        n = QP.numvar
        m = QP.numeq
        mi = QP.numineq
        self.x = copy(QP.x0)

        self.czl = spmatrix([],[],[],(mi,1)) 
        self.czu = spmatrix([],[],[],(mi,1)) 

        self.y   = matrix(1.0,(m,1))

        self.zl  = spmatrix([],[],[],(n,1))
        self.zu  = spmatrix([],[],[],(n,1))
        #self.zl  = spmatrix([0],[max(0,n-1)],[0])
        #self.zu  = spmatrix([0],[max(0,n-1)],[0])

        self.iter = 0
        self.cAL = []
        self.cAU = []
        self.cI = []
        self.AL = []
        self.AU = []
        self.I = []

        self.violations = Violations()


        self.option = ut.optimset(**kwargs)
        self._guess()
        self._ObserverList = dict(PDAS.observerlist)

        self.state = 'Initial'
        self._CGiter = 0
        self.TotalCG = 0
        self.CG_r = matrix([10,100,1000])
        self.correctV = Violations()
        if 'inv_norm' in self.option.options.keys():
            self.inv_norm = self.option['inv_norm']
        else:
            self.inv_norm = 10

    @property
    def cgiter(self):
        # Access _cgiter
        return self._CGiter

    @cgiter.setter
    def cgiter(self,value):
        # Set _cgiter, will triger total cgiter to accumulate
        self._CGiter = value
        self.TotalCG += value
        
    def _get_lenV(self):
        p = self.violations
        return len(p.Vxl + p.Vxu+p.Vxcl+p.Vxcu+p.Vzl+p.Vzu+p.Vzcl+p.Vzcu)

    lenV = property(_get_lenV)
    # def _get_CG_r(self):
    #     return self._CG_r

    # def _set_CG_r(self,val):
    #     self._CG_r = val

    # CG_r = property(_get_CG_r,_set_CG_r)

    def _update_observerlist(self,**kwargs):
        'Batch change/override observerlist'
        self._ObserverList.update(kwargs)

    def register(self,obstype,observer):
        'Register an observer to a observertype '
        dic = self._ObserverList
        obslist = dic.get(obstype)
        if obslist or obslist == []:
            # Observer type exist: append observer
            obslist.append(observer)
        else:
            # Observer type nonexist: create the type and append observer
            obslist[obstype] = [observer]

    def unregister(self,obstype,observer):
        'Unregister an observer from a observertype '
        obslist = self._ObserverList.get(obstype)
        if obslist:
            # Observer type exist: remove the element
            obslist.remove(observer)


    def notify(self,obstypelist, what = {}):
        'Notify obserers what to do'
        for obstype in obstypelist:
            sublist = self._ObserverList.get(obstype)
            if sublist:
                for observer in sublist:
                    observer(what)
            else:
                pass
                #print('{0} is not registered to the observerlist!'.format(obstype))

    def ask(self,observer, what = {}):
        'Ask obserers, return its response'
        sublist = self._ObserverList.get(observer)
        if sublist:
            for observer in sublist:
                return observer(what)
            else:
                print('{0} is not registered to the observerlist!'.format(obstype))


    def exact_solve(self):
        'Solve the attached problem by exact solver'
        # Attach necessary observers
        p = obs.printer(self)
        self.register('printer',p)
        col = obs.collector(self)
        self.register('collector',col)

        # Main loop
        while self.iter < self.option['MaxItr']:
            # Do subspace minimization on current partition

            self.ssm()
            # Identify which indices are violated
            self.violations = self.identify_violation()

            self.notify(['printer','collector'])

            # Optimality check
            if self.kkt < self.option['OptTol']:
                self.state = 'Optimal'
                break
            
            # Obtain a new partition
            self.newp()
            self.iter += 1

        # When finished running
        print '-'*80
        print 'Problem Status          :', self.state
        print 'Total Iterations        :', self.iter
        print 'Total Krylov-iterations :', self.TotalCG
        print 'Avg norm(r)/norm(r0    ): {0:<.2e}'.format(col.res_relative/col.num)
        print 'Avg norm(r)             : {0:<.2e}'.format(col.res_abs/col.num)
        print 'Time Elapsed            : {0:<.2e}'.format(col.time_elapse)
        print '-'*80+'\n\n'

        self.unregister('printer',p)
        self.unregister('collector',col)
        return (self.x,self.iter,self.TotalCG,col.res_relative/col.num,col.res_abs/col.num,col.time_elapse,self.state)

    def inexact_solve(self):
        'Solve the attached problem by exact solver'
        # Attach necessary observers, does not work!!
        self._ObserverList['printer'] = []
        p = obs.printer(self)
        k = obs.monitor(self,est_fun = self.option['fun_estinv'])
        self.register('printer',p)
        self.register('monitor',k)

        c = obs.conditioner(self)
        self.register('conditioner',c)
        col = obs.collector(self)
        self.register('collector',col)
        # Initialize necessary structure

        # Main loop


        while self.iter < self.option['MaxItr']:
            # Fix active primals and inactive duals
            self._fix()

            # Let the LinsysWrap calculate an inexact solution and modify PDAS
            L = LinsysWrap(self,minres)
            L.solve()

            # Notify observers about this iteration
            self.cgiter = L.iter
            self.notify(['monitor','collector','printer'])
            # Optimality check
            if self.kkt < self.option['OptTol']:
                self.state = 'Optimal'
                break
            
            # Obtain a new partition
            self.newp(by = self.correctV)
            self.iter += 1

        # When finished running

        # Unregister the conditioner, otherwise may affact next call

        print '-'*80
        print 'Problem Status          :', self.state
        print 'Total Iterations        :', self.iter
        print 'Total Krylov-iterations :', self.TotalCG
        print 'Avg norm(r)/norm(r0)    : {0:<.2e}'.format(col.res_relative/col.num)
        print 'Avg norm(r)             : {0:<.2e}'.format(col.res_abs/col.num)
        print 'Time Elapsed            : {0:<.2e}'.format(col.time_elapse)
        print 'Total Krylov4estimation :', k.cgiter
        print '-'*80+'\n\n'

        self.unregister('printer',p)
        self.unregister('conditioner',c)
        self.unregister('monitor',k)
        self.unregister('collector',col)

        return (self.x,self.iter,self.TotalCG,col.res_relative/col.num,col.res_abs/col.num,col.time_elapse,self.state,k.cgiter)

#        self.unregister('conditioner',c)
#        self.unregister('collector',col)

    def identify_violation(self, by = None):
        'Find violated sets'
        qp = self.QP
        if by is None:
            by = self
        # Violated sets, or estimated violation sets for inexact case
        p = Violations()
        Vxl  = pick_negative(by.x[self.I] - qp.l[self.I])[1]
        p.Vxl  = [self.I[i] for i in Vxl]
        Vxu  = pick_negative(qp.u[self.I] - by.x[self.I])[1]
        p.Vxu  = [self.I[i] for i in Vxu]
        Vxcl = pick_negative(qp.A[self.cI,:]*by.x - qp.bl[self.cI])[1]
        p.Vxcl = [self.cI[i] for i in Vxcl]
        Vxcu = pick_negative(qp.bu[self.cI] - qp.A[self.cI,:]*by.x)[1]
        p.Vxcu = [self.cI[i] for i in Vxcu]
        Vzl  = pick_negative(by.zl[self.AL])[1]
        p.Vzl  = [self.AL[i] for i in Vzl]
        Vzu  = pick_negative(by.zu[self.AU])[1]
        p.Vzu  = [self.AU[i] for i in Vzu]
        Vzcl = pick_negative(by.czl[self.cAL])[1]
        p.Vzcl = [self.cAL[i] for i in Vzcl]
        Vzcu = pick_negative(by.czu[self.cAU])[1]
        p.Vzcu = [self.cAU[i] for i in Vzcu]

        return p

    def identify_violation_inexact(self,lb,ub):
        'Find correctly identified violation sets from the error bounds'
        # Estimate violation
        self.violations = self.identify_violation()

        # Obtain sizes
        nI = len(self.I)
        ny = self.QP.numeq
        ncL = len(self.cAL)
        ncU = len(self.cAU)

        # Bound for some variables
        xI_lb = self.x[self.I] + lb[0:nI]
        xI_ub = self.x[self.I] + ub[0:nI]

        czl_cAL_ub = []
        czu_cAU_ub = []
        if self.czl.size[0] > 0:
            czl_cAL_ub = self.czl[self.cAL] + ub[nI+ny:nI+ny+ncL]
            czu_cAU_ub = self.czu[self.cAU] + ub[nI+ny+ncL:]

        # Error bounds for some variables
        err_xI_lb = lb[0:nI]
        err_xI_ub = ub[0:nI]
        

        qp = self.QP
        eq_err_zl = sparse([[qp.H[self.AL,self.I]], [qp.Aeq[:,self.AL].T], [-qp.A[self.cAL,self.AL].T],[qp.A[self.cAU,self.AL].T]])
        eq_err_zu = -sparse([[qp.H[self.AU,self.I]], [qp.Aeq[:,self.AU].T], [-qp.A[self.cAL,self.AU].T],[qp.A[self.cAU,self.AU].T]])
        eq_err_Ax = sparse([qp.A[self.cI,self.I]])

        # Bound for other variables
        zlAL_ub = self.zl[self.AL] + estimate_range(eq_err_zl,lb,ub)[1]
        zuAU_ub = self.zu[self.AU] + estimate_range(eq_err_zu,lb,ub)[1]
        err_Ax_lb, err_Ax_ub = estimate_range(eq_err_Ax,err_xI_lb,err_xI_ub)

        Ax_cI_lb = self.Ax[self.cI] + err_Ax_lb
        Ax_cI_ub = self.Ax[self.cI] + err_Ax_ub

        # Generate the correctly identified violated sets
        correct = self.correctV
        Vxl  = pick_negative(xI_ub - qp.l[self.I])[1]
        Vxu  = pick_negative(qp.u[self.I] - xI_lb)[1]

        Vxcl = pick_negative(Ax_cI_ub - qp.bl[self.cI])[1]
        Vxcu = pick_negative(qp.bu[self.cI] - Ax_cI_lb)[1]
        Vzl  = pick_negative(zlAL_ub)[1]
        Vzu  = pick_negative(zuAU_ub)[1]
        Vzcl = pick_negative(czl_cAL_ub)[1]
        Vzcu = pick_negative(czu_cAU_ub)[1]


        correct.Vxl  = [self.I[i] for i in Vxl]
        correct.Vxu  = [self.I[i] for i in Vxu]
                
        correct.Vxcl = [self.cI[i] for i in Vxcl]
        correct.Vxcu = [self.cI[i] for i in Vxcu]
        correct.Vzl  = [self.AL[i] for i in Vzl]
        correct.Vzu  = [self.AU[i] for i in Vzu]
        correct.Vzcl = [self.cAL[i] for i in Vzcl]
        correct.Vzcu = [self.cAU[i] for i in Vzcu]


        lenV = len(correct.Vxl + correct.Vxu+correct.Vxcl+correct.Vxcu+correct.Vzl+correct.Vzu+correct.Vzcl+correct.Vzcu)
        return lenV

    def ssm(self):
        'Do subspace minimization'
        # Fix primal active, dual in-active variables
        self._fix()
        # Obtain and solve the lienar equation
        self._solve_lin()
        # Back substitute to get other unknowns
        self._back_substitute()

    def _fix(self):
        'Auxilliary function: fix primal active and dual inactive variables'
        # Fix primal and dual variables
        self.x[self.AL] = self.QP.l[self.AL]
        self.x[self.AU] = self.QP.u[self.AU]
#        if len(self.I+self.AU) > 0:
        self.zl[self.I+self.AU] = 0
#        if len(self.I+self.AL) > 0:

        self.zu[self.I+self.AL] = 0

        if self.czl.size[0] > 0:
            self.czl[self.cI+self.cAU] = 0
            self.czu[self.cI+self.cAL] = 0

    def _solve_lin(self):
        'Auxilliary function: solve the linear system exactly'
        # Linear equation coefficients
        Lhs, rhs, x0 = self._get_lineq()
        # Set x[I], y, czl, czu by solving linear equation
        # xy = cholmod.splinsolve(Lhs,rhs)

        self.CG_r0 = Lhs*x0 - rhs

        # Convert to numpy/scipy matrix
        npLhs = cvxopt_to_numpy_matrix(Lhs)
        nprhs = cvxopt_to_numpy_matrix(matrix(rhs))
        npx0 = cvxopt_to_numpy_matrix(x0)

        # Solve the linear system
        collector = Iter_collect(tol=self.option['ResTol'])
#        collector = Iter_collect()
        cg = minres(npLhs,nprhs,npx0,callback=collector)
        xy = numpy_to_cvxopt_matrix(cg[0])
        self.cgiter = collector.iter
        self.CG_r = Lhs*numpy_to_cvxopt_matrix(collector.x) - rhs

        nI = len(self.I)
        ny = self.QP.numeq
        ncL = len(self.cAL)
        ncU = len(self.cAU)
        self.x[self.I] = xy[0:nI]
        if xy[nI:nI + ny].size[1] == 1:
            self.y = xy[nI:nI + ny]

        if self.czl.size[0] > 0:
            self.czl[self.cAL] = xy[nI+ny:nI+ny+ncL]
            self.czu[self.cAU] = xy[nI+ny+ncL:]

    def _back_substitute(self):
        'Auxilliary function: back substitute after solving the linear system'
        # Back substitute to obtain zl, zu
        qp = self.QP
        eq1 = qp.H*self.x + qp.c

        if len(qp.Aeq)!=0:
            eq1 = eq1 + qp.Aeq.T*self.y
        if len(self.cAL + self.cAU)!=0:
            eq1 = eq1 + qp.A.T*(self.czu - self.czl)

        self.zl[self.AL] = eq1[self.AL]
        self.zu[self.AU] = -eq1[self.AU]
        if len(qp.A) != 0:
            self.Ax = qp.A*self.x
        else:
            self.Ax = spmatrix([],[],[],(0,1))


    def newp(self,by = None):
        'Obtain a new partition by identifing violations'
        if by is None:
            by = self.violations

        # Move violations
        self.I   = setdiff(self.I,union(by.Vxl,by.Vxu))
        self.cI  = setdiff(self.cI,union(by.Vxcl,by.Vxcu))
        self.I   = union(self.I,union(by.Vzl,by.Vzu))
        self.cI  = union(self.cI,union(by.Vzcl,by.Vzcu))
        self.AL  = union(setdiff(self.AL,by.Vzl),by.Vxl)
        self.AU  = union(setdiff(self.AU,by.Vzu),by.Vxu)
        self.cAL = union(setdiff(self.cAL,by.Vzcl), by.Vxcl)
        self.cAU = union(setdiff(self.cAU,by.Vzcu), by.Vxcu)

    def _guess(self,x0 = None):
        'Guess the initial active-set from x0'
        if not x0:
            x0 = self.QP.x0
        eps = self.option['eps']
        self.AL = pick_negative(x0 - self.QP.l - eps)[1]
        self.AU = pick_negative(self.QP.u - x0 - eps)[1]

        self.I = setdiff(range(self.QP.numvar),union(self.AL,self.AU))

        if self.QP.A.size[0] > 0:
            self.cAL = pick_negative(self.QP.A*x0 - self.QP.bl - eps)[1]
            self.cAU = pick_negative(self.QP.bu -self.QP.A*x0 - eps)[1]

    @property
    def kkt(self):
        'Access the value of kkt'
        return self._compute_kkt()



    def _compute_kkt(self):
        'Compute the optimality error vector, complementarity not included'
        qp = self.QP
        # KKT equations
        eq1 = qp.H*self.x + qp.c - self.zl + self.zu

        if len(qp.Aeq)!=0:
            eq1 = eq1 + qp.Aeq.T*self.y
        if len(self.cAL + self.cAU)!=0:
            eq1 = eq1 + qp.A.T*(self.czu - self.czl)
        
        eq2 = qp.Aeq*self.x - qp.beq
        # Primal and dual violations
        Ax = qp.A*self.x
        feas_Ax_l = matrix(pick_negative(Ax - qp.bl)[0])
        feas_Ax_u = matrix(pick_negative(qp.bu - Ax)[0])
        feas_Ax   = matrix([feas_Ax_l,feas_Ax_u])
        feas_x_l  = matrix(pick_negative(self.x - qp.l)[0])
        feas_x_u  = matrix(pick_negative(qp.u - self.x)[0])
        feas_x    = matrix([feas_x_l,feas_x_u])
        feas_zl   = matrix(filter(lambda x: x < 0 ,self.zl))
        feas_zu   = matrix(filter(lambda x: x < 0 ,self.zu))
#        pdb.set_trace()
        return np.linalg.norm(matrix([eq1,eq2,feas_Ax,feas_x,feas_zl,feas_zu]),np.inf)

    def _compute_obj(self):
        'Compute the objective function value'
        qp = self.QP
        obj = 0.5*self.x.T*qp.H*self.x + qp.c.T*self.x
        return obj[0]

    def _get_lineq(self):
        'Extract linear equation coefficients from QP and a partition'
        qp = self.QP
        # Concatenate active constraints with eq constraints

        Aeq = sparse([qp.Aeq,-qp.A[self.cAL,:],qp.A[self.cAU,:]])
        beq = matrix([qp.beq,-qp.bl[self.cAL],qp.bu[self.cAU]])
        beq -= Aeq[:,self.AL]*qp.l[self.AL] + Aeq[:,self.AU]*qp.u[self.AU]
        Aeq = Aeq[:,self.I]
        # [H(I,I);Aeq(:,I)]
        Lhs = sparse([qp.H[self.I,self.I], Aeq ])
        row_Aeq, col_Aeq = Aeq.size
        # [Aeq(:,I)';zeros]
        if row_Aeq != 0:
            AeqT0 = sparse([Aeq.T,spmatrix([0],[row_Aeq-1],[row_Aeq-1])])
        else:
            AeqT0 = Aeq.T
        # Concatenate by collumn
        Lhs = sparse([[Lhs],[AeqT0]])
        # Concatenat to yield rhs
        rhs = sparse(matrix([-qp.c[self.I] -qp.H[self.I,self.AL]*qp.l[self.AL] -qp.H[self.I,self.AU]*qp.u[self.AU] ,beq]))

        x0 = matrix([self.x[self.I],self.y,self.czl[self.cAL],self.czu[self.cAU]])
        return (Lhs,rhs,x0)

class PDASc(PDAS):

    def __init__(self,prob = QP(),Free=[],**kwargs):
        'Initialize the PDASc'
        # Initialize the superclass
        super(PDASc,self).__init__(prob,**kwargs)
        # Add additional data
        self.F = Free
        nf = len(self.F)
        m = self.QP.numeq
        Q1 = sparse([self.QP.H[self.F,self.F], self.QP.Aeq[:,self.F] ])
        Q2 = sparse([self.QP.Aeq[:,self.F].T, spmatrix([],[],[],(m,m)) ])

        self.ipiv = matrix(1,(nf+m,1))
        self.Q = matrix([[Q1],[Q2]])
        lapack.sytrf(self.Q,self.ipiv)
        # LDL factorization of Q, dense, option 1
        # lapack.sytrf(self.Q,self.ipiv)
        # self.DiagQi = spdiag([self.Q[i,i] for i in range(nf+m)])
        # self.LQ = copy(self.Q)
        # for i in range(nf+m):
        #     self.LQ[i,i] = 1
        #     for j in range(i+1,nf+m):
        #         self.LQ[i,j] = 0

        # LDL factorization of Q, dense, option 2 by calling Matlab library
        # mlab = Matlab()
        # mlab.start()
        filename = 'temp/'+''.join([random.choice(string.ascii_letters + string.digits) for n in xrange(32)])
        s = dict()
        s['A'] = sparse([[Q1],[Q2]])
        write(filename+'.mat',s)
        d = '/home/zhh210/workspace/pypdas/numeric/control/'
        # Memory failure when size is large
        # output = mlab.run_func('getldp.m',{'arg1':filename})

        # self.LQ = matrix(output['result']['L'])
        # self.Dinv = matrix(output['result']['Dinv'])
        # self.P = matrix(output['result']['P'])
        # output = mlab.run_func('saveldp.m',{'arg1':filename})

        # Execute shell command
        cmd = 'matlab -r '+'"'+ "saveldp('"+filename + "');quit" + '"'
        print cmd
        os.system(cmd)
        data = read(filename+'.mat')
        self.LQ = matrix(data['L'])
        self.Dinv = data['Dinv']
        self.P = data['P']
        #mlab.stop()
        

    def inexact_solve(self,dynamic = False):
        'Solve the attached problem by exact solver'
        # Attach necessary observers, does not work!!
        self._ObserverList['printer'] = []
        p = obs.printer(self)
        k = obs.monitor(self,est_fun = self.option['fun_estinv'])
        self.register('printer',p)
        if dynamic:
            self.register('monitor',k)

        c = obs.conditioner(self)
        self.register('conditioner',c)
        col = obs.collector(self)
        self.register('collector',col)
        # Initialize necessary structure

        # Main loop


        while self.iter < self.option['MaxItr']:
            # Fix active primals and inactive duals
            self._fix()

            # Let the LinsysWrap calculate an inexact solution and modify PDAS
            L = LinsysWrap_c(self,minres,dynamic)
            L.solve()

            # Notify observers about this iteration
            self.cgiter = L.iter
            self.notify(['collector','printer','monitor'])
            # Optimality check
            if self.kkt < self.option['OptTol']:
                self.state = 'Optimal'
                break
            
            # Obtain a new partition
            self.newp(by = self.correctV)
            self.iter += 1

        # When finished running

        # Unregister the conditioner, otherwise may affact next call

        print '-'*80
        print 'Problem Status          :', self.state
        print 'Total Iterations        :', self.iter
        print 'Total Krylov-iterations :', self.TotalCG
        print 'Avg norm(r)/norm(r0)    : {0:<.2e}'.format(col.res_relative/col.num)
        print 'Avg norm(r)             : {0:<.2e}'.format(col.res_abs/col.num)
        print 'Time Elapsed            : {0:<.2e}'.format(col.time_elapse)
        print 'Total Krylov4estimation :', col.poweriter
        print '-'*80+'\n\n'

        self.unregister('printer',p)
        self.unregister('conditioner',c)
        self.unregister('collector',col)

        return (self.x,self.iter,self.TotalCG,col.res_relative/col.num,col.res_abs/col.num,col.time_elapse,self.state,col.poweriter)



    def identify_violation_inexact_c(self,lb,ub,SaQinvSi):
        'Find correctly identified violation sets from the error bounds'
        # Estimate violation
        self.violations = self.identify_violation()

        # Obtain sizes
        nI = len(self.realI)
        ny = self.QP.numeq
        ncL = len(self.cAL)
        ncU = len(self.cAU)

        # Bound for some variables
        xI_lb = self.x[self.realI] + lb
        xI_ub = self.x[self.realI] + ub

        czl_cAL_ub = []
        czu_cAU_ub = []
        if self.czl.size[0] > 0:
            czl_cAL_ub = self.czl[self.cAL] + ub[nI+ny:nI+ny+ncL]
            czu_cAU_ub = self.czu[self.cAU] + ub[nI+ny+ncL:]

        # Error bounds for some variables
        err_xI_lb = lb
        err_xI_ub = ub

        qp = self.QP
#        eq_err_zl = sparse([[qp.H[self.AL,self.realI]], [qp.Aeq[:,self.AL].T], [-qp.A[self.cAL,self.AL].T],[qp.A[self.cAU,self.AL].T]])
        eq_err_zu = sparse(-qp.H[self.AU,self.realI] + SaQinvSi)
#        eq_err_Ax = sparse([qp.A[self.cI,self.realI]])

        # Bound for other variables
#        zlAL_ub = self.zl[self.AL] + estimate_range(eq_err_zl,lb,ub)[1]
        zuAU_ub = self.zu[self.AU] + estimate_range(eq_err_zu,lb,ub)[1]
#        err_Ax_lb, err_Ax_ub = estimate_range(eq_err_Ax,err_xI_lb,err_xI_ub)
#        Ax_cI_lb = self.Ax[self.cI] + err_Ax_lb
#        Ax_cI_ub = self.Ax[self.cI] + err_Ax_ub
 
        # Generate the correctly identified violated sets
        correct = self.correctV
        Vxl  = [] #pick_negative(xI_ub - qp.l[self.realI])[1]
        Vxu  = pick_negative(qp.u[self.realI] - xI_lb)[1]

        Vxcl = [] #pick_negative(Ax_cI_ub - qp.bl[self.cI])[1]
        Vxcu = [] #pick_negative(qp.bu[self.cI] - Ax_cI_lb)[1]
        Vzl  = [] #pick_negative(zlAL_ub)[1]
        Vzu  = pick_negative(zuAU_ub)[1]
        Vzcl = [] #pick_negative(czl_cAL_ub)[1]
        Vzcu = [] #pick_negative(czu_cAU_ub)[1]


        correct.Vxl  = [self.realI[i] for i in Vxl]
        correct.Vxu  = [self.realI[i] for i in Vxu]
                
        correct.Vxcl = [self.cI[i] for i in Vxcl]
        correct.Vxcu = [self.cI[i] for i in Vxcu]
        correct.Vzl  = [self.AL[i] for i in Vzl]
        correct.Vzu  = [self.AU[i] for i in Vzu]
        correct.Vzcl = [self.cAL[i] for i in Vzcl]
        correct.Vzcu = [self.cAU[i] for i in Vzcu]


        lenV = len(correct.Vxl + correct.Vxu+correct.Vxcl+correct.Vxcu+correct.Vzl+correct.Vzu+correct.Vzcl+correct.Vzcu)
        return lenV

    def _get_lineq_c(self):
        'Extract linear equation coefficients from QP and a partition'
        qp = self.QP

        # Inactive but not free
        self.realI = setdiff(self.I,self.F)

        # Put in the order of [realI,F]
        pI = self.realI + self.F

        # Concatenate active constraints with eq constraints

        Aeq = sparse([qp.Aeq,-qp.A[self.cAL,:],qp.A[self.cAU,:]])
        beq = matrix([qp.beq,-qp.bl[self.cAL],qp.bu[self.cAU]])
        beq -= Aeq[:,self.AL]*qp.l[self.AL] + Aeq[:,self.AU]*qp.u[self.AU]
        Aeq = Aeq[:,pI]
        # [H(I,I);Aeq(:,I)]
        Lhs = sparse([qp.H[pI,pI], Aeq ])
        row_Aeq, col_Aeq = Aeq.size
        # [Aeq(:,I)';zeros]
        if row_Aeq != 0:
            AeqT0 = sparse([Aeq.T,spmatrix([0],[row_Aeq-1],[row_Aeq-1])])
        else:
            AeqT0 = Aeq.T
        # Concatenate by collumn
        Lhs = sparse([[Lhs],[AeqT0]])
        # Concatenat to yield rhs
        rhs = sparse(matrix([-qp.c[pI] -qp.H[pI,self.AL]*qp.l[self.AL] -qp.H[pI,self.AU]*qp.u[self.AU] ,beq]))

        x0 = matrix([self.x[pI],self.y,self.czl[self.cAL],self.czu[self.cAU]])
        return (Lhs,rhs,x0)


    def test():
        print 'PDASs inherited from PDAS.'

def pick_negative(x):
    'An auxilliary function to find positions of negative(violation)'
    tups = zip(*[(i,j) for i,j in enumerate(x) if j < 0])
    if len(tups) == 0:
        tups = [[],[]]
    else:
        tups = [list(tups[1]),list(tups[0])]

    # If x is in sparse format
    if hasattr(x,'I'):
        tups[1] = list(x.I[tups[1]])
    return tups    


class Iter_collect(object):
    'A auxilliary function object to collect number of iterations'
    def __init__(self,tol = 1.0e-8):
        self.iter = 0
        self.x = None
        self.tol = tol

    def __call__(self,xk):
        frame = inspect.currentframe().f_back
        self.iter = frame.f_locals['itn']
        self.x = xk
        A = frame.f_locals['A']
        b = frame.f_locals['b']

        if np.linalg.norm(A*self.x-b,np.inf) < self.tol:
            # Reaches optimal tolerance
            set_in_frame(frame,'istop',9)
        else:
            set_in_frame(frame,'istop',0)


def set_in_frame(frame, name, value):
    frame.f_locals[name] = value
    locals_to_fast(frame, 1)


def blockinv(M):
    'Function to compute inverse of 1x1 or 2x2 block matrix'
    i = 0
    while i < M.size[0]-1:
        if M[i,i+1] != 0:
            tmp = M[i,i]
            M[i:i+1,i:i+1] /= M[i,i]*M[i+1,i+1] - M[i,i+1]*M[i+1,i]
            M[i,i] = M[i+1,i+1]
            M[i+1,i+1] = tmp
            M[i,i+1] *= -1
            M[i+1,i] *= -1
            i += 1
        else:
            M[i,i] = 1/M[i,i]
        i += 1
    if i == M.size[0] -1 and M[i-1,i] != 0:
        M[i,i] = 1/M[i,i]
    return M

def _test_pdas():
    'Test function of pdas'
    print 'testing function _test_pdas()\n'
    qp = randQP(100)
    pdas = PDAS(qp)
    #p = obs.printer(pdas)
    #pdas.register('printer',p)
    #c = obs.conditioner(pdas)
    #pdas.register('conditioner',c)
    #pdas.ssm()
    #pdas.newp()
    pdas2 = copy(pdas)
    pdas.exact_solve()
    pdas2.inexact_solve()

def _test_pick():
    'Test function of pick_negative()'
    a = spmatrix([-1],[3],[0])
    val, ind = pick_negative(a)
    print val, ind

if __name__ == '__main__':
    _test_pdas()

