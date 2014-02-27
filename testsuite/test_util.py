from utility import optimset


def test_optimset():
    'Test function of OptOptions'
    print 'Default options:'
    option = optimset()
    for key,item in option.options.iteritems():
        print key, item
    print 'Set options: Solver=pdas, NewOp=1'
    option = optimset(Solver='pdas',NewOp=1)
    for key in option.options.keys():
        print key, option[key]

if __name__=='__main__':
    test_optimset()
