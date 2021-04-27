from FADO import *
import SU2
import scipy.optimize
import subprocess
import numpy as np
from scipy.optimize import minimize
from externalrunextension import *
from sqpoptimizer import *


config = SU2.io.Config("naca0012_config_original.cfg")
designparams = copy.deepcopy(config['DV_VALUE_OLD'])

var = InputVariable(np.array(designparams),ArrayLabelReplacer("__X__"))

pType_direct = Parameter(["DIRECT"],LabelReplacer("__MATH_PROBLEM__"))
pType_adjoint = Parameter(["DISCRETE_ADJOINT"],LabelReplacer("__MATH_PROBLEM__"))
pType_mesh_filename_original = Parameter(["mesh_NACA0012_inv.su2"],LabelReplacer("__MESH_FILENAME__"))
pType_mesh_filename_deformed = Parameter(["mesh_NACA0012_inv_def.su2"],LabelReplacer("__MESH_FILENAME__"))

pType_hessian_active = Parameter(["YES"],LabelReplacer("__ACTIVATE_HESSIAN__"))
pType_hessian_passive = Parameter(["NO"],LabelReplacer("__ACTIVATE_HESSIAN__"))

pType_ObjFun_DRAG = Parameter(["DRAG"],LabelReplacer("__OBJECTIVE_FUNCTION__"))
pType_ObjFun_LIFT = Parameter(["LIFT"],LabelReplacer("__OBJECTIVE_FUNCTION__"))

pType_Iter_run = Parameter(["10"],LabelReplacer("__NUM_ITER__"))
pType_Iter_step = Parameter(["1"],LabelReplacer("__NUM_ITER__"))

pType_OneShot_passive = Parameter(["NONE"], LabelReplacer("__ONESHOT_MODE__"))
pType_OneShot_active = Parameter(["PIGGYBACK"], LabelReplacer("__ONESHOT_MODE__"))

### FOR MESH DEFORMATION ###
meshDeformationRun = SU2MeshDeformationSkipFirstIteration("DEFORM","mpirun -n 4 SU2_DEF naca0012_config_tmpl.cfg",True,"naca0012_config_tmpl.cfg")
meshDeformationRun.addConfig("naca0012_config_tmpl.cfg")
meshDeformationRun.addData("mesh_NACA0012_inv.su2")
meshDeformationRun.addParameter(pType_direct)
meshDeformationRun.addParameter(pType_Iter_run)
meshDeformationRun.addParameter(pType_mesh_filename_original)
meshDeformationRun.addParameter(pType_ObjFun_DRAG) #not actually needed, but used to make a valid config file
meshDeformationRun.addParameter(pType_hessian_passive)
meshDeformationRun.addParameter(pType_OneShot_passive)
### END # FOR MESH DEFORMATION ###

### FOR FLOW AND ADJOINT SOLUTION ###
directRun = ExternalSU2CFDOneShotSingleZoneDriverWithRestartOption("DIRECT","mpirun -n 4 SU2_CFD_AD naca0012_config_tmpl.cfg",True,"naca0012_config_tmpl.cfg")
directRun.addConfig("naca0012_config_tmpl.cfg")
directRun.addData("DEFORM/mesh_NACA0012_inv_def.su2")
directRun.addData("solution_flow.dat") #has to be an actual solution file
directRun.addData("solution_adj_cd.dat") #dummy adj solution file
directRun.addParameter(pType_adjoint)
directRun.addParameter(pType_Iter_run)
directRun.addParameter(pType_mesh_filename_deformed)
directRun.addParameter(pType_ObjFun_DRAG)
directRun.addParameter(pType_hessian_active)
directRun.addParameter(pType_OneShot_active)
### END # FOR FLOW AND ADJOINT SOLUTION ###

### FOR LIFT CONSTRAINT ADJOINT ###
adjointRunLift = ExternalSU2CFDDiscAdjSingleZoneDriverWithRestartOption("ADJOINT_LIFT","mpirun -n 4 SU2_CFD_AD naca0012_config_tmpl.cfg",True,"naca0012_config_tmpl.cfg")
adjointRunLift.addConfig("naca0012_config_tmpl.cfg")
adjointRunLift.addData("DEFORM/mesh_NACA0012_inv_def.su2")
adjointRunLift.addData("DIRECT/solution_flow.dat")
adjointRunLift.addData("solution_adj_cl.dat") #dummy adj solution file
adjointRunLift.addParameter(pType_adjoint)
adjointRunLift.addParameter(pType_Iter_run)
adjointRunLift.addParameter(pType_mesh_filename_deformed)
adjointRunLift.addParameter(pType_ObjFun_LIFT)
adjointRunLift.addParameter(pType_hessian_passive)
adjointRunLift.addParameter(pType_OneShot_passive)

dotProductRunLift = ExternalRun("DOT_LIFT","mpirun -n 4 SU2_DOT_AD naca0012_config_tmpl.cfg",True)
dotProductRunLift.addConfig("naca0012_config_tmpl.cfg")
dotProductRunLift.addData("DEFORM/mesh_NACA0012_inv_def.su2")
dotProductRunLift.addData("ADJOINT_LIFT/solution_adj_cl.dat")
dotProductRunLift.addParameter(pType_adjoint)
dotProductRunLift.addParameter(pType_Iter_run)
dotProductRunLift.addParameter(pType_mesh_filename_deformed)
dotProductRunLift.addParameter(pType_ObjFun_LIFT)
dotProductRunLift.addParameter(pType_hessian_passive)
dotProductRunLift.addParameter(pType_OneShot_passive)
### END # FOR LIFT CONSTRAINT ###

### Define Function and Constraints out of the runs ###
fun = Function("DRAG","DIRECT/history.csv",TableReader(0,0,start=(-1,12),end=(None,None),delim=","))
fun.addInputVariable(var,"DIRECT/orig_grad.dat",TableReader(0,None,start=(0,0),end=(None,None),delim=","))
fun.addValueEvalStep(meshDeformationRun)
fun.addValueEvalStep(directRun)

liftConstraint = Function("LIFT","DIRECT/history.csv",TableReader(0,0,start=(-1,13),end=(None,None),delim=","))
liftConstraint.addInputVariable(var,"DOT_LIFT/of_grad.dat",TableReader(None,0,start=(1,0),end=(None,None)))
liftConstraint.addValueEvalStep(meshDeformationRun)
liftConstraint.addValueEvalStep(directRun)
liftConstraint.addGradientEvalStep(adjointRunLift)
liftConstraint.addGradientEvalStep(dotProductRunLift)
### END # Define Function and Constraints out of the runs ###

# Driver
driver = ScipyDriver()

def_objs = config['OPT_OBJECTIVE']
this_obj = def_objs.keys()[0]
sign  = SU2.io.get_objectiveSign(this_obj)
driver.addObjective("min", fun, sign)

driver.addEquality(liftConstraint, 0.33, 1.0)

### Postprocess command ###
directSolutionFilename = "DIRECT/solution_flow.dat"
pathForDirectSolutionFilename = os.path.join(driver._workDir,directSolutionFilename)
commandDirectSolution = "cp" + " " + pathForDirectSolutionFilename + " ."
print("command 1: ", commandDirectSolution)

adjointSolutionDRAG = "DIRECT/solution_adj_cd.dat"
pathForAdjointSolutionDRAG = os.path.join(driver._workDir,adjointSolutionDRAG)
commandAdjointSolutionDRAG = "cp" + " " + pathForAdjointSolutionDRAG + " ."
print("command 2: ", commandAdjointSolutionDRAG)
driver.setUserPostProcessGrad(commandDirectSolution + "&&" + commandAdjointSolutionDRAG + "&&" + "echo $PWD")

adjointSolutionLIFT = "ADJOINT_LIFT/solution_adj_cl.dat"
pathForAdjointSolutionLIFT = os.path.join(driver._workDir,adjointSolutionLIFT)
commandAdjointSolutionLIFT = "cp" + " " + pathForAdjointSolutionLIFT + " ."
print("command 3: ", commandAdjointSolutionLIFT)
driver.setUserPostProcessEqConGrad(commandAdjointSolutionLIFT + "&&" + "echo $PWD/DESIGN")
### END # Postprocess command ###

driver.preprocess()
driver.setEvaluationMode(False)
driver.setStorageMode(True,"DESIGN/DSN_")

log = open("log.txt","w",1)
his = open("history.txt","w",1)
driver.setLogger(log)
driver.setHistorian(his)

x  = driver.getInitial()

maxIter = int (config.OPT_ITERATIONS)           # number of opt iterations
bound_upper = float (config.OPT_BOUND_UPPER)    # variable bound to be scaled by the line search
bound_lower = float (config.OPT_BOUND_LOWER)    # variable bound to be scaled by the line search

accu = float(config.OPT_ACCURACY)       # optimizer accuracy

mode = float(config.LINESEARCH_MODE)    # linesearch mode

xb_low = [float(bound_lower)]*driver._nVar      # lower dv bound it includes the line search acceleration factor
xb_up  = [float(bound_upper)]*driver._nVar      # upper dv bound it includes the line search acceleration fa
xbounds = list(zip(xb_low, xb_up)) # design bounds

# scale accuracy
eps = 1.0e-04

driver.setConstraintGradientEvalMode(False)

driver.hessian_eval_parameters("DIRECT", "of_hess.dat")

conf = RSQPconfig()

outputs = SQPconstrained(x0=x,
                         func=driver.fun,
                         f_eqcons= driver.eq_cons,
                         f_ieqcons= empty_func,
                         fprime=driver.grad,
                         fprime_eqcons= driver.eq_cons_grad,
                         fprime_ieqcons= empty_func,
                         fdotdot= driver.hess,
                         iter=maxIter,
                         acc=accu,
                         lsmode=mode,
                         config=conf,
                         xb=xbounds,
                         driver=driver)

log.close()
his.close()
  
print ('Finished')
