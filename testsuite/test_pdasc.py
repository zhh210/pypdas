from pdas.pdas import PDASc
from prob import randQP
from copy import copy

def test_pdas():
    'Test function of pdasc, subclass of pdas'
    print 'testing function _test_pdas()\n'
    qp = randQP(100)
    pdas = PDASc(qp)
    #p = obs.printer(pdas)
    #pdas.register('printer',p)
    #c = obs.conditioner(pdas)
    #pdas.register('conditioner',c)
    #pdas.ssm()
    #pdas.newp()
    pdas2 = copy(pdas)
    pdas.exact_solve()
    pdas2.inexact_solve()


if __name__=='__main__':
    test_pdas()
