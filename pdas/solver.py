'''
Interface taht defines the functionality of a solver.
'''
from prob import QP


class solver(object):
    'An abstract solver class served as an unified interface.'
    def __init__(self,prob = QP(),**kwargs):
        # Initialize the solver
        self.prob = prob
        super().__init__()
        pass

    def solve(self):
        # Solve the problem attached
        pass


