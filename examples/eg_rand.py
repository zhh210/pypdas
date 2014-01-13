'''
Examples demonstrating generating random matrix, random QP, etc.
'''
from randutil import sprandsym, sp_rand
from prob import randQP, randBQP
from pdas.pdas import PDAS
from copy import copy

# Generate a random sparse matrix with 3 rows, 4 columns, sparsity 0.8
mat = sp_rand(3,4,0.8)
print(mat)

# Generate a random positive definite matrix with 5 variables, 
# condition number 100, sparsity 0.5. Note: random Jocobi rotations
# are applied.
pdmat = sprandsym(size=5,cond=100,sp=0.5)
print(pdmat)

# Generate a random QP with 100 variables, 1 equality, 1 inequaity
qp1 = randQP(size = 100, numeq = 1, numineq = 1)
qp2 = randQP(size = 100, numeq = 1, numineq = 1)

# Generate a random bound constrained QP with 100 variables
bqp1 = randBQP(size = 100) 

# Another way to generate a random bound constrained QP with 100 variables
bqp2 = randQP(size = 100, numeq = 0, numineq = 0) 


# Solve the random bound constrained QP by PDAS with exact solve
pdas = PDAS(bqp1)
pdas.exact_solve()

# Solve the random bound constrained QP by PDAS with inexact solve
# import pdb
# pdb.set_trace()
pdas2 = PDAS(bqp2)
pdas2.inexact_solve()


# Solve the random bound constrained QP by PDAS with exact solve
pdas3 = PDAS(qp1)
pdas3.exact_solve()

# Solve the random bound constrained QP by PDAS with inexact solve
pdas4 = PDAS(qp2)
pdas4.inexact_solve()
