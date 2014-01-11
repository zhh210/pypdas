
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
            print '='*95
            print '   Iter       Obj         KKT          |AL|    |AU|    |I|     |V|     |cV|   invnorm  CG-iter'
            print '='*95
        print '{iter:>6d}  {obj:^12.2e}  {kkt:^12.2e}  {AL:>6d}  {AU:>6d}  {I:>6d}  {V:>6d}  {cV:>6d}   {inv:>6.2e}  {CGiter:>6d}'.format(iter=p.iter,obj=p._compute_obj(), kkt=p.kkt,AL=len(p.AL),AU=len(p.AU),I=len(p.I),V=p.lenV,cV=p.correctV.lenV,inv=p.inv_norm,CGiter=p.cgiter)

class conditioner(observer):
    'Conditioner class'
    option = {
        'CG_res_absolute' : 1.0e-16,
        'CG_res_relative' : 1.0e-3,
        'identified_estimated_ratio' : 0.8
        }
    def __init__(self,iPDAS,**kwarg):
        'Attach conditioner to solver iPDAS'
        self.solver = iPDAS
        self.option = dict(conditioner.option)
        self.option.update(kwarg)

    def __call__(self,what = {}):
        'Return whether the condition is satisfied'
        satisfied = (
            (not self.enforce_exact())
             and (self.is_CG_res_absolute() 
             or self.is_CG_res_relative()
             or self.is_identified_estimate()
             or self.all_identified())
        )

        return satisfied

    def is_CG_res_absolute(self):
        'Linear solver has satisfied absolute tolerance'
        return max(abs(self.solver.CG_r)) < self.option['CG_res_absolute']
    
    def is_CG_res_relative(self):
        'Linear solver has satisfied relative tolerance'
        return max(abs(self.solver.CG_r))/max(abs(self.solver.CG_r0)) < self.option['CG_res_relative']

    def is_identified_estimate(self):
        '|identified V|/|estimated V| sufficiently large'
        lenIdentified = 0
        lenEstimated = 0
        for key,val in self.solver.correctV.__dict__.iteritems():
            lenIdentified += len(val)
            lenEstimated += len(getattr(self.solver.violations,key))
        #print lenIdentified, lenEstimated
        return lenIdentified > self.option['identified_estimated_ratio']*lenEstimated

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

        

class monitor(observer):
    '''Monitor class'''
    pass

class collector(observer):
    '''Collector class'''
    pass
