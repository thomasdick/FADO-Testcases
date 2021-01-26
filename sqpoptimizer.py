#!/usr/bin/env python
# This Python file uses the following encoding: utf-8

## \file sqpoptimizer.py
#  \brief Python script for performing the SQP optimization with a semi handwritten optimizer.
#  \author T. Dick

import sys
import numpy as np
import cvxopt
import csv


def SQPconstrained(x0, func, f_eqcons, f_ieqcons, fprime, fprime_eqcons, fprime_ieqcons, fdotdot, iter, acc, lsmode, xb=None):
    """ This is a implementation of a SQP optimizer
        It is written for smoothed derivatives
        Accepts:
            - equality and inequality constraints
            - approximated hessians
            - additional parameters
        can be applied to analytical functions for test purposes"""

    sys.stdout.write('Using the hand implemented version. Setting up the environment...' + '\n')

    # start by evaluating the the functions
    p = x0
    F = func(p)
    E = f_eqcons(p)
    C = f_ieqcons(p)
    D_F = fprime(p)
    D_E = fprime_eqcons(p)
    D_C = fprime_ieqcons(p)
    H_F = fdotdot(p)

    # set the inout parameters
    err = 2*acc+1
    step = 1

    # extract design parameter bounds
    if xb is None:
        xb = [[-1e-1]*len(p), [1e-1]*len(p)]
    else:
        #unzip the list
        xb = list(zip(*xb))

    # prepare output, using 'with' command to automatically close the file in case of exceptions
    with open("optimizer_history.csv", "w") as outfile:
        csv_writer = csv.writer(outfile, delimiter=',')
        header = ['iter', 'objective function', 'equal constraint', 'inequal constraint', 'design', 'norm(gradient)', 'norm(delta_p)', 'lm_eqcons', 'lm_ieqcons', 'Lagrangian1', 'Lagrangian2','gradLagrangian1','gradLagrangian2']
        csv_writer.writerow(header)

        # main optimizer loop
        while (err > acc and step <= iter):

            sys.stdout.write('Optimizer iteration: ' + str(step) + '\n')

            if step > 1:
                # reevaluate the functions
                F = func(p)
                E = f_eqcons(p)
                C = f_ieqcons(p)
                D_F = fprime(p)
                D_E = fprime_eqcons(p)
                D_C = fprime_ieqcons(p)
                H_F = fdotdot(p)

            # assemble equality constraints
            if np.size(E) > 0:
                A = cvxopt.matrix(D_E)
                b = cvxopt.matrix(-E)

            # expand inequality constraints by bounds
            Id = np.identity(len(p))
            if np.size(C) > 0:
                G = cvxopt.matrix(np.block([[-D_C], [-Id], [Id]]))
                h = cvxopt.matrix(np.append(C,np.append( -np.array(xb[0]), np.array(xb[1]))))
            else:
                G = cvxopt.matrix(np.block([[-Id], [Id]]))
                h = cvxopt.matrix(np.append( -np.array(xb[0]), np.array(xb[1])))

            # pack objective function
            P = cvxopt.matrix(H_F)
            q = cvxopt.matrix(D_F)

            # solve the interior quadratic problem
            if np.size(E) > 0:
                sol = cvxopt.solvers.qp(P, q, G, h, A, b)
            else:
                sol = cvxopt.solvers.qp(P, q, G, h)

            # line search
            delta_p = np.array([i for i in sol['x']])
            delta_p = linesearch(p, delta_p, F, func, E, f_eqcons, 0, 0, acc, lsmode)

            # update the design
            p = p + delta_p
            err = np.linalg.norm(delta_p, 2)

            sys.stdout.write('New design: ' + str(p) + '\n')

            # extract some values from the quadratic solver
            lm_eqcons = np.array([i for i in sol['y']])
            lm_ieqcons = np.zeros(np.size(C))
            for i in range(np.size(C)):
                lm_ieqcons[i] = sol['z'][i]

            Lagrangian1 = F + np.inner(E,lm_eqcons)
            Lagrangian2 = Lagrangian1 + np.inner(C,lm_ieqcons)
            gradL1 = D_F + lm_eqcons @ D_E
            gradL2 = gradL1 + lm_ieqcons @ D_C

            # write to the history file
            line = [step, F, E, C, p, np.linalg.norm(D_F, 2), err, lm_eqcons, lm_ieqcons, Lagrangian1, Lagrangian2, gradL1, gradL2]
            csv_writer.writerow(line)
            outfile.flush()

            # increase counter at the end of the loop
            step += 1

    # end output automatically

    return 0

# end SQPconstrained


def SQPequalconstrained(x0, func, f_eqcons, fprime, fprime_eqcons, fdotdot, iter, acc, lsmode, xb=None):
    """ This is a implementation of a SQP optimizer
        It is written for smoothed derivatives
        Accepts:
            - only equality constraints
            - approximated hessians
            - additional parameters
        can be applied to analytical functions for test purposes"""

    sys.stdout.write('Using the simplified hand implemented version for equality constraints. Setting up the environment...' + '\n')

    # start by evaluating the the functions
    p = x0
    F = func(p)
    E = f_eqcons(p)
    D_F = fprime(p)
    D_E = fprime_eqcons(p)
    H_F = fdotdot(p)

    # set the inout parameters
    nu = np.sign(E)
    err = 2*acc+1
    step = 1

    # compute the Lagrangian
    L = F + np.dot(nu, E)

    # prepare output, using 'with' command to automatically close the file in case of exceptions
    with open("optimizer_history.csv", "w") as outfile:
        csv_writer = csv.writer(outfile, delimiter=',')
        header = ['iter', 'objective function', 'equal constraint', 'Lagrange multiplier', 'design', 'norm(gradient)', 'norm(delta_p)']
        csv_writer.writerow(header)

        # main optimizer loop
        while (err > acc and step <= iter):

            sys.stdout.write('Optimizer iteration: ' + str(step) + '\n')

            if step > 1:
                # reevaluate the functions
                F = func(p)
                E = f_eqcons(p)
                D_F = fprime(p)
                D_E = fprime_eqcons(p)
                H_F = fdotdot(p)
                L = F + np.dot(nu, E)

            sys.stdout.write('objective function: ' + str(F) +
                             ' , equality constrain: ' + str(E) +
                             ' , Lagrangian: ' + str(L) + '\n')

            # assemble linear equation system
            rhs = np.append([-D_F], [-E])
            mat = np.block([[H_F, D_E.T], [D_E, 0]])

            # solve the LES
            sol = np.linalg.solve(mat, rhs)

            # FIND way to incorporate boundary
            # G = cvxopt.matrix(np.block([[-Id], [Id]]))
            # h = cvxopt.matrix(np.append([-xb[0]]*len(p), [xb[1]]*len(p)))

            # get the solution
            delta_p = sol[0:len(p)]
            nu_temp = sol[-np.size(nu):]

            # line search
            delta_p = linesearch(p, delta_p, F, func, E, f_eqcons, nu, nu_temp, acc, lsmode)

            # update the design
            p = p + delta_p
            nu = nu_temp
            err = np.linalg.norm(delta_p, 2)

            sys.stdout.write('Current design: ' + str(p) + ' , Lagrangian multiplier: ' + str(nu) + '\n')

            # write to the history file
            line = [step, F, E, nu, p, np.linalg.norm(D_F, 2), err]
            csv_writer.writerow(line)
            outfile.flush()

            # increase counter at the end of the loop
            step += 1

    # end output automatically

    return 0

# end SQPequalconstrained


def linesearch(p, delta_p, F, func, E, f_eqcons, nu_old, nu_new, acc, lsmode):

    mode = lsmode

    # use a constant reduction factor
    if (mode >= 0.0):
        delta_p = mode*delta_p
        sys.stdout.write("descend step: " + str(delta_p) + "\n")

    # backtracking based on objective function
    elif (mode == -1.0):
        criteria = True
        while (criteria):
            p_temp = p + delta_p
            sys.stdout.write("descend step: " + str(delta_p) + "\n")
            F_new = func(p_temp)
            # test for optimization progress
            if (F_new > F):
                sys.stdout.write("not a reduction in objective function. \n")
                delta_p = 0.5*delta_p
            else:
                sys.stdout.write("descend step accepted. \n")
                criteria = False

            #avoid step getting to small
            if (np.linalg.norm(delta_p, 2) < acc):
                sys.stdout.write("can't find a good step. \n")
                criteria = False

    # backtracking based on the Laplacian
    elif (mode == -2.0):
        criteria = True
        while (criteria):
            p_temp = p + delta_p
            F_new = func(p_temp)
            E_new = f_eqcons(p_temp)
            L_new = F_new + np.dot(nu_new, E_new)
            sys.stdout.write("new function: " + str(F_new) + " , new constraint: " + str(E_new) + " , new Lagrangian: " + str(L_new) + "\n")
            # test for optimization progress
            if (L_new > (F+nu_old*E)):
                sys.stdout.write("not a reduction in Lagrangian function. \n")
                delta_p = 0.5*delta_p
            else:
                sys.stdout.write("descend step accepted. \n")
                criteria = False

            #avoid step getting to small
            if (np.linalg.norm(delta_p, 2) < acc):
                sys.stdout.write("can't find a good step. \n")
                criteria = False

    # if unknown mode choosen, leave direction unaffected.
    else:
        sys.stdout.write("Unknown line search mode. leave search direction unaffected. \n")
        sys.stdout.write("descend step: " + str(delta_p) + "\n")

    return delta_p

# end of linesearch


# we need a unit matrix to have a dummy for optimization tests.
def unit_hessian(x):
    return np.identity(np.size(x))
# end of unit_hessian


# we need an empty function call to have a dummy for optimization tests.
def empty_func(x):
    return []
# end of empty_func
