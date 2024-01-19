
import numpy as np

class SplineBasis:
    
    def __init__(self, degree, knots):
        self._degree = degree
        self._knots = np.array(sorted(knots))



def lazy_test():
    basis = SplineBasis(3, [3,1,2,0])
    print(basis._knots)

lazy_test()
