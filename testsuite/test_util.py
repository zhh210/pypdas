from utility import OptOptions


def test_OptOptions():
    'Test function of OptOptions'
    print 'Default options:'
    option = OptOptions()
    for key,item in option.options.iteritems():
        print key, item
    print 'Set options: Solver=pdas, NewOp=1'
    option = OptOptions(Solver='pdas',NewOp=1)
    for key in option.options.keys():
        print key, option[key]

if __name__=='__main__':
    test_OptOptions()
