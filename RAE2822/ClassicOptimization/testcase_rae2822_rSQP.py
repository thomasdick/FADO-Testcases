from FADO import *
import SU2
import scipy.optimize
import subprocess
import numpy as np
from scipy.optimize import minimize
from externalrunextension import *
from sqpoptimizer import *

### DEFINE CONFIG, MESH, CONFIG LABELS #
config = SU2.io.Config("optimization_RAE2822_orig.cfg")
designparams = copy.deepcopy(config['DV_VALUE_OLD'])
var = InputVariable(np.array(designparams),ArrayLabelReplacer("__X__"))

pType_FDstep = Parameter(["0.001"+",0.001"*(len(designparams)-1)], LabelReplacer("__X__"))

pType_direct = Parameter(["DIRECT"],LabelReplacer("__MATH_PROBLEM__"))
pType_adjoint = Parameter(["DISCRETE_ADJOINT"],LabelReplacer("__MATH_PROBLEM__"))
pType_mesh_filename_original = Parameter(["rae2822_ffd.su2"],LabelReplacer("__MESH_FILENAME__"))
pType_mesh_filename_deformed = Parameter(["rae2822_ffd_def.su2"],LabelReplacer("__MESH_FILENAME__"))

pType_Iter_run = Parameter(["10000"],LabelReplacer("__NUM_ITER__"))
pType_Iter_step = Parameter(["1"],LabelReplacer("__NUM_ITER__"))
pType_hessian_active = Parameter(["YES"],LabelReplacer("__ACTIVATE_HESSIAN__"))
pType_hessian_passive = Parameter(["NO"],LabelReplacer("__ACTIVATE_HESSIAN__"))

pType_ObjFun_DRAG = Parameter(["DRAG"],LabelReplacer("__OBJECTIVE_FUNCTION__"))
pType_ObjFun_LIFT = Parameter(["LIFT"],LabelReplacer("__OBJECTIVE_FUNCTION__"))
pType_ObjFun_MOMENT_Z = Parameter(["MOMENT_Z"],LabelReplacer("__OBJECTIVE_FUNCTION__"))
### END # DEFINE CONFIG, MESH, CONFIG LABELS #


### FOR MESH DEFORMATION ###
meshDeformationRun = SU2MeshDeformationSkipFirstIteration("DEFORM","mpirun -n 4 SU2_DEF optimization_RAE2822_tmpl.cfg",True,"optimization_RAE2822_tmpl.cfg")
meshDeformationRun.addConfig("optimization_RAE2822_tmpl.cfg")
meshDeformationRun.addData("rae2822_ffd.su2")
meshDeformationRun.addParameter(pType_direct)
meshDeformationRun.addParameter(pType_Iter_run)
meshDeformationRun.addParameter(pType_mesh_filename_original)
meshDeformationRun.addParameter(pType_ObjFun_DRAG) #not actually needed, but used to make a valid config file
meshDeformationRun.addParameter(pType_hessian_passive)
### END # FOR MESH DEFORMATION ###

### FOR FLOW SOLUTION ###
directRun = ExternalSU2CFDSingleZoneDriverWithRestartOption("DIRECT","mpirun -n 4 SU2_CFD optimization_RAE2822_tmpl.cfg",True,"optimization_RAE2822_tmpl.cfg")
directRun.addConfig("optimization_RAE2822_tmpl.cfg")
directRun.addData("DEFORM/rae2822_ffd_def.su2")
directRun.addData("solution_flow.dat") #dummy solution file
directRun.addParameter(pType_direct)
directRun.addParameter(pType_Iter_run)
directRun.addParameter(pType_mesh_filename_deformed)
directRun.addParameter(pType_ObjFun_DRAG)
directRun.addParameter(pType_hessian_passive)
### END # FOR FLOW SOLUTION ###

### FOR DRAG OBJECTIVE ADJOINT ###
adjointRunDrag = ExternalSU2CFDDiscAdjSingleZoneDriverWithRestartOption("ADJOINT_DRAG","mpirun -n 4 SU2_CFD_AD optimization_RAE2822_tmpl.cfg",True,"optimization_RAE2822_tmpl.cfg")
adjointRunDrag.addConfig("optimization_RAE2822_tmpl.cfg")
adjointRunDrag.addData("DEFORM/rae2822_ffd_def.su2")
adjointRunDrag.addData("DIRECT/solution_flow.dat")
adjointRunDrag.addData("solution_adj_cd.dat") #dummy adj soluion file
adjointRunDrag.addParameter(pType_adjoint)
adjointRunDrag.addParameter(pType_Iter_run)
adjointRunDrag.addParameter(pType_mesh_filename_deformed)
adjointRunDrag.addParameter(pType_ObjFun_DRAG)
adjointRunDrag.addParameter(pType_hessian_passive)

dotProductRunDrag = ExternalRun("DOT_DRAG","mpirun -n 4 SU2_DOT_AD optimization_RAE2822_tmpl.cfg",True)
dotProductRunDrag.addConfig("optimization_RAE2822_tmpl.cfg")
dotProductRunDrag.addData("DEFORM/rae2822_ffd_def.su2")
dotProductRunDrag.addData("ADJOINT_DRAG/solution_adj_cd.dat")
dotProductRunDrag.addParameter(pType_adjoint)
dotProductRunDrag.addParameter(pType_Iter_run)
dotProductRunDrag.addParameter(pType_mesh_filename_deformed)
dotProductRunDrag.addParameter(pType_ObjFun_DRAG)
dotProductRunDrag.addParameter(pType_hessian_passive)
### END # FOR DRAG OBJECTIVE ADJOINT ###

### FOR DRAG OBJECTIVE HESSIAN ###
hessianRunDrag = ExternalSU2CFDDiscAdjSingleZoneDriverWithRestartOption("HESSIAN","mpirun -n 4 SU2_CFD_AD optimization_RAE2822_tmpl.cfg",True,"optimization_RAE2822_tmpl.cfg")
hessianRunDrag.addConfig("optimization_RAE2822_tmpl.cfg")
hessianRunDrag.addData("DEFORM/rae2822_ffd_def.su2")
hessianRunDrag.addData("DIRECT/solution_flow.dat")
hessianRunDrag.addData("ADJOINT_DRAG/solution_adj_cd.dat")
hessianRunDrag.addParameter(pType_adjoint)
hessianRunDrag.addParameter(pType_Iter_step)
hessianRunDrag.addParameter(pType_mesh_filename_deformed)
hessianRunDrag.addParameter(pType_ObjFun_DRAG)
hessianRunDrag.addParameter(pType_hessian_active)
### END # FOR DRAG OBJECTIVE HESSIAN ###

### FOR LIFT CONSTRAINT ADJOINT ###
adjointRunLift = ExternalSU2CFDDiscAdjSingleZoneDriverWithRestartOption("ADJOINT_LIFT","mpirun -n 4 SU2_CFD_AD optimization_RAE2822_tmpl.cfg",True,"optimization_RAE2822_tmpl.cfg")
adjointRunLift.addConfig("optimization_RAE2822_tmpl.cfg")
adjointRunLift.addData("DEFORM/rae2822_ffd_def.su2")
adjointRunLift.addData("DIRECT/solution_flow.dat")
adjointRunLift.addData("solution_adj_cl.dat") #dummy adj soluion file
adjointRunLift.addParameter(pType_adjoint)
adjointRunLift.addParameter(pType_Iter_run)
adjointRunLift.addParameter(pType_mesh_filename_deformed)
adjointRunLift.addParameter(pType_ObjFun_LIFT)
adjointRunLift.addParameter(pType_hessian_passive)

dotProductRunLift = ExternalRun("DOT_LIFT","mpirun -n 4 SU2_DOT_AD optimization_RAE2822_tmpl.cfg",True)
dotProductRunLift.addConfig("optimization_RAE2822_tmpl.cfg")
dotProductRunLift.addData("DEFORM/rae2822_ffd_def.su2")
dotProductRunLift.addData("ADJOINT_LIFT/solution_adj_cl.dat")
dotProductRunLift.addParameter(pType_adjoint)
dotProductRunLift.addParameter(pType_Iter_run)
dotProductRunLift.addParameter(pType_mesh_filename_deformed)
dotProductRunLift.addParameter(pType_ObjFun_LIFT)
dotProductRunLift.addParameter(pType_hessian_passive)
### END # FOR LIFT CONSTRAINT ###

### FOR MOMENTUM CONSTRAINT ADJOINT ###
adjointRunMomZ = ExternalSU2CFDDiscAdjSingleZoneDriverWithRestartOption("ADJOINT_MOMENT_Z","mpirun -n 4 SU2_CFD_AD optimization_RAE2822_tmpl.cfg",True,"optimization_RAE2822_tmpl.cfg")
adjointRunMomZ.addConfig("optimization_RAE2822_tmpl.cfg")
adjointRunMomZ.addData("DEFORM/rae2822_ffd_def.su2")
adjointRunMomZ.addData("DIRECT/solution_flow.dat")
adjointRunMomZ.addData("solution_adj_cmz.dat") #dummy adj soluion file
adjointRunMomZ.addParameter(pType_adjoint)
adjointRunMomZ.addParameter(pType_Iter_run)
adjointRunMomZ.addParameter(pType_mesh_filename_deformed)
adjointRunMomZ.addParameter(pType_ObjFun_MOMENT_Z)
adjointRunMomZ.addParameter(pType_hessian_passive)

dotProductRunMomZ = ExternalRun("DOT_MOMENT_Z","mpirun -n 4 SU2_DOT_AD optimization_RAE2822_tmpl.cfg",True)
dotProductRunMomZ.addConfig("optimization_RAE2822_tmpl.cfg")
dotProductRunMomZ.addData("DEFORM/rae2822_ffd_def.su2")
dotProductRunMomZ.addData("ADJOINT_MOMENT_Z/solution_adj_cmz.dat")
dotProductRunMomZ.addParameter(pType_adjoint)
dotProductRunMomZ.addParameter(pType_Iter_run)
dotProductRunMomZ.addParameter(pType_mesh_filename_deformed)
dotProductRunMomZ.addParameter(pType_ObjFun_MOMENT_Z)
dotProductRunMomZ.addParameter(pType_hessian_passive)
### END # FOR MOMENTUM CONSTRAINT ###

### FOR GEOMETRY CONSTRAINT ###
geometryFDRun = ExternalRun("GEOMETRY", "mpirun -n 4 SU2_GEO optimization_RAE2822_tmpl.cfg",True)
geometryFDRun.addConfig("optimization_RAE2822_tmpl.cfg")
geometryFDRun.addData("DEFORM/rae2822_ffd_def.su2")
geometryFDRun.addParameter(pType_FDstep)
geometryFDRun.addParameter(pType_direct)
geometryFDRun.addParameter(pType_Iter_run)
geometryFDRun.addParameter(pType_mesh_filename_deformed)
geometryFDRun.addParameter(pType_ObjFun_DRAG)
geometryFDRun.addParameter(pType_hessian_passive)
### END # FOR GEOMETRY CONSTRAINT ###

### Define Function and Constraints out of the runs ###
fun = Function("DRAG","DIRECT/history.csv",TableReader(0,0,start=(-1,8),end=(None,None),delim=","))
fun.addInputVariable(var,"DOT_DRAG/of_grad.dat",TableReader(None,0,start=(1,0),end=(None,None)))
fun.addValueEvalStep(meshDeformationRun)
fun.addValueEvalStep(directRun)
fun.addGradientEvalStep(adjointRunDrag)
fun.addGradientEvalStep(dotProductRunDrag)
fun.addGradientEvalStep(hessianRunDrag)

liftConstraint = Function("LIFT","DIRECT/history.csv",TableReader(0,0,start=(-1,9),end=(None,None),delim=","))
liftConstraint.addInputVariable(var,"DOT_LIFT/of_grad.dat",TableReader(None,0,start=(1,0),end=(None,None)))
liftConstraint.addValueEvalStep(meshDeformationRun)
liftConstraint.addValueEvalStep(directRun)
liftConstraint.addGradientEvalStep(adjointRunLift)
liftConstraint.addGradientEvalStep(dotProductRunLift)

momentConstraint = Function("MOMENT_Z","DIRECT/history.csv",TableReader(0,0,start=(-1,13),end=(None,None),delim=","))
momentConstraint.addInputVariable(var,"DOT_MOMENT_Z/of_grad.dat",TableReader(None,0,start=(1,0),end=(None,None)))
momentConstraint.addValueEvalStep(meshDeformationRun)
momentConstraint.addValueEvalStep(directRun)
momentConstraint.addGradientEvalStep(adjointRunMomZ)
momentConstraint.addGradientEvalStep(dotProductRunMomZ)

thicknessConstraint = Function("GEOMETRY","GEOMETRY/of_func.csv",TableReader(0,0,start=(-1,1),end=(None,None),delim=","))
thicknessConstraint.addInputVariable(var,"GEOMETRY/of_grad.csv",TableReader(None,0,start=(1,2),end=(None,None),delim=","))
thicknessConstraint.addValueEvalStep(meshDeformationRun)
thicknessConstraint.addValueEvalStep(geometryFDRun)
thicknessConstraint.addGradientEvalStep(geometryFDRun)
### END # Define Function and Constraints out of the runs ###

# Driver
driver = ScipyDriver()

def_objs = config['OPT_OBJECTIVE']
this_obj = def_objs.keys()[0]
sign  = SU2.io.get_objectiveSign(this_obj)
driver.addObjective("min", fun, sign)

driver.addEquality(liftConstraint, 0.724, 1.0)
driver.addUpperBound(momentConstraint, 0.093, 1.0)
driver.addLowerBound(thicknessConstraint, 0.12, 1.0)

### Postprocess command ###
directSolutionFilename = "DIRECT/solution_flow.dat"
pathForDirectSolutionFilename = os.path.join(driver._workDir,directSolutionFilename)
commandDirectSolution = "cp" + " " + pathForDirectSolutionFilename + " ."
print("command 1: ", commandDirectSolution) 

adjointSolutionDRAG = "ADJOINT_DRAG/solution_adj_cd.dat"
pathForAdjointSolutionDRAG = os.path.join(driver._workDir,adjointSolutionDRAG)
commandAdjointSolutionDRAG = "cp" + " " + pathForAdjointSolutionDRAG + " ."
print("command 2: ", commandAdjointSolutionDRAG)
driver.setUserPostProcessGrad(commandDirectSolution + " && " + commandAdjointSolutionDRAG)

adjointSolutionLIFT = "ADJOINT_LIFT/solution_adj_cl.dat"
pathForAdjointSolutionLIFT = os.path.join(driver._workDir,adjointSolutionLIFT)
commandAdjointSolutionLIFT = "cp" + " " + pathForAdjointSolutionLIFT + " ."
print("command 3: ", commandAdjointSolutionLIFT)
driver.setUserPostProcessEqConGrad(commandAdjointSolutionLIFT)

adjointSolutionMOMZ = "ADJOINT_MOMENT_Z/solution_adj_cmz.dat"
pathForAdjointSolutionMOMZ = os.path.join(driver._workDir,adjointSolutionMOMZ)
commandAdjointSolutionMOMZ = "cp" + " " + pathForAdjointSolutionMOMZ + " ."
print("command 4: ", commandAdjointSolutionMOMZ)
driver.setUserPostProcessIEqConGrad(commandAdjointSolutionMOMZ)
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
eps = 1.0e-010

driver.setConstraintGradientEvalMode(False)

driver.hessian_eval_parameters("HESSIAN", "of_hess.dat")

conf = RSQPconfig()

outputs = SQPconstrained(x0=x,
                         func=driver.fun,
                         f_eqcons= driver.eq_cons,
                         f_ieqcons= driver.ieq_cons,
                         fprime=driver.grad,
                         fprime_eqcons= driver.eq_cons_grad,
                         fprime_ieqcons= driver.ieq_cons_grad,
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




