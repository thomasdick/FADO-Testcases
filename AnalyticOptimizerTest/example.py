# This example tests the sqp optimizer
#
#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

import sys
import numpy as np

import importlib.util
spec = importlib.util.spec_from_file_location("module.name", "/home/tdick/Documents/FadoFramework/FADO/sqpoptimizer.py")
sqp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sqp)

class Parameter:
    "Dummy to test behaviour of SU2 project class."
    def __init__( self, other_config ):
        self.config = other_config

def parabel(p):
    return (p[0] - 1)**2 + p[1]**2

def parabel_jacobian(p):
    return np.array([ 2.0*p[0]-2.0, 2*p[1]])

def parabel_hessian(p):
    return np.block([[ 2.0, 0.0], [ 0.0, 2.0]])

def empty_func(p):
    return np.zeros([0])

def empty_grad(p):
    return np.zeros([0,len(p)])

def equal(p):
    return np.block([4-p[0]-p[1], p[0]-1.5])

def equal_jacobian(p):
    return np.block([[-1.0, -1.0], [1.0, 0.0]])

def inequal(p):
    return np.block([-2*p[0]+3*p[1]-1, p[0]-1.5])

def inequal_jacobian(p):
    return np.block([[-2.0, 3.0], [1.0, 0.0]])


def main():
    "main routine"

    sys.stdout.write("I have a trivial example function.")

    p = [0.0, 0.0]
    mode = -2.0
    bound = [-1e2, 1e2]
    xb_low = [-1e2]*2
    xb_up  = [1e2]*2
    bound = list(zip(xb_low, xb_up))

    sys.stdout.write("f(x,y)= " + str(parabel(p)) + "\n")
    sys.stdout.write("Grad f(x,y)= " + str(parabel_jacobian(p)) + "\n")
    sys.stdout.write("Hess f(x,y)= " + str(parabel_hessian(p)) + "\n")

    outputs = sqp.SQPequalconstrained( x0 = p, func = parabel, f_eqcons = equal, fprime = parabel_jacobian, fprime_eqcons = equal_jacobian, fdotdot = parabel_hessian, iter = 10, acc = 1e-8, lsmode=mode, xb = bound)

#    outputs = sqp.SQPconstrained( x0 = p,
#                                  func = parabel, f_eqcons = equal, f_ieqcons = inequal,
#                                  fprime = parabel_jacobian, fprime_eqcons = equal_jacobian, fprime_ieqcons = inequal_jacobian,
#                                  fdotdot = parabel_hessian,
#                                  iter = 10, acc = 1e-8, lsmode=mode, xb = bound)

if __name__ == "__main__":
     main()
