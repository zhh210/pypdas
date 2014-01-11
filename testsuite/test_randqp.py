from pdas.pdas import PDAS
from prob import randQP
from copy import copy

def test_pdas():
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


if __name__=='__main__':
    test_pdas()
