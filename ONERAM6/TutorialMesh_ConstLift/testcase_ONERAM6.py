from FADO import *
import SU2
import scipy.optimize
import subprocess
import numpy as np
from scipy.optimize import minimize
from externalrunextension import *
from sqpoptimizer import *

### DEFINE CONFIG, MESH, CONFIG LABELS #
config = SU2.io.Config("config_Optimization_orig.cfg")
designparams = copy.deepcopy(config['DV_VALUE_OLD'])
var = InputVariable(np.array(designparams),ArrayLabelReplacer("__X__"))

pType_FDstep = Parameter(["0.001"+",0.001"*(len(designparams)-1)], LabelReplacer("__X__"))

pType_direct = Parameter(["DIRECT"],LabelReplacer("__MATH_PROBLEM__"))
pType_adjoint = Parameter(["DISCRETE_ADJOINT"],LabelReplacer("__MATH_PROBLEM__"))
pType_mesh_filename_original = Parameter(["mesh_tutorial_ffd.su2"],LabelReplacer("__MESH_FILENAME__"))
pType_mesh_filename_deformed = Parameter(["mesh_tutorial_ffd_deform.su2"],LabelReplacer("__MESH_FILENAME__"))

pType_Iter_run = Parameter(["7500"],LabelReplacer("__NUM_ITER__"))
pType_Iter_step = Parameter(["1"],LabelReplacer("__NUM_ITER__"))
pType_hessian_active = Parameter(["YES"],LabelReplacer("__ACTIVATE_HESSIAN__"))
pType_hessian_passive = Parameter(["NO"],LabelReplacer("__ACTIVATE_HESSIAN__"))

pType_ObjFun_DRAG = Parameter(["DRAG"],LabelReplacer("__OBJECTIVE_FUNCTION__"))
### END # DEFINE CONFIG, MESH, CONFIG LABELS #


### FOR MESH DEFORMATION ###
meshDeformationRun = SU2MeshDeformationSkipFirstIteration("DEFORM","mpirun -n 1 SU2_DEF config_Optimization_tmpl.cfg",True,"config_Optimization_tmpl.cfg")
meshDeformationRun.addConfig("config_Optimization_tmpl.cfg")
meshDeformationRun.addData("mesh_tutorial_ffd.su2")
meshDeformationRun.addParameter(pType_direct)
meshDeformationRun.addParameter(pType_Iter_run)
meshDeformationRun.addParameter(pType_mesh_filename_original)
meshDeformationRun.addParameter(pType_ObjFun_DRAG) #not actually needed, but used to make a valid config file
meshDeformationRun.addParameter(pType_hessian_passive)
### END # FOR MESH DEFORMATION ###

### FOR FLOW SOLUTION ###
directRun = ExternalSU2CFDSingleZoneDriverWithRestartOption("DIRECT","mpirun -n 4 SU2_CFD config_Optimization_tmpl.cfg",True,"config_Optimization_tmpl.cfg")
directRun.addConfig("config_Optimization_tmpl.cfg")
directRun.addData("DEFORM/mesh_tutorial_ffd_deform.su2")
directRun.addData("solution_flow.dat") #dummy solution file
directRun.addParameter(pType_direct)
directRun.addParameter(pType_Iter_run)
directRun.addParameter(pType_mesh_filename_deformed)
directRun.addParameter(pType_ObjFun_DRAG)
directRun.addParameter(pType_hessian_passive)
### END # FOR FLOW SOLUTION ###

### FOR DRAG OBJECTIVE ADJOINT ###
adjointRunDrag = ExternalSU2CFDDiscAdjSingleZoneDriverWithRestartOption("ADJOINT_DRAG","mpirun -n 4 SU2_CFD_AD config_Optimization_tmpl.cfg",True,"config_Optimization_tmpl.cfg")
adjointRunDrag.addConfig("config_Optimization_tmpl.cfg")
adjointRunDrag.addData("DEFORM/mesh_tutorial_ffd_deform.su2")
adjointRunDrag.addData("DIRECT/solution_flow.dat")
adjointRunDrag.addData("solution_adj_cd.dat") #dummy adj soluion file
adjointRunDrag.addParameter(pType_adjoint)
adjointRunDrag.addParameter(pType_Iter_run)
adjointRunDrag.addParameter(pType_mesh_filename_deformed)
adjointRunDrag.addParameter(pType_ObjFun_DRAG)
adjointRunDrag.addParameter(pType_hessian_passive)

dotProductRunDrag = ExternalRun("DOT_DRAG","mpirun -n 4 SU2_DOT_AD config_Optimization_tmpl.cfg",True)
dotProductRunDrag.addConfig("config_Optimization_tmpl.cfg")
dotProductRunDrag.addData("DEFORM/mesh_tutorial_ffd_deform.su2")
dotProductRunDrag.addData("ADJOINT_DRAG/solution_adj_cd.dat")
dotProductRunDrag.addParameter(pType_adjoint)
dotProductRunDrag.addParameter(pType_Iter_run)
dotProductRunDrag.addParameter(pType_mesh_filename_deformed)
dotProductRunDrag.addParameter(pType_ObjFun_DRAG)
dotProductRunDrag.addParameter(pType_hessian_passive)
### END # FOR DRAG OBJECTIVE ADJOINT ###

### FOR DRAG OBJECTIVE HESSIAN ###
hessianRunDrag = ExternalSU2CFDDiscAdjSingleZoneDriverWithRestartOption("HESSIAN","mpirun -n 4 SU2_CFD_AD config_Optimization_tmpl.cfg",True,"config_Optimization_tmpl.cfg")
hessianRunDrag.addConfig("config_Optimization_tmpl.cfg")
hessianRunDrag.addData("DEFORM/mesh_tutorial_ffd_deform.su2")
hessianRunDrag.addData("DIRECT/solution_flow.dat")
hessianRunDrag.addData("ADJOINT_DRAG/solution_adj_cd.dat")
hessianRunDrag.addParameter(pType_adjoint)
hessianRunDrag.addParameter(pType_Iter_step)
hessianRunDrag.addParameter(pType_mesh_filename_deformed)
hessianRunDrag.addParameter(pType_ObjFun_DRAG)
hessianRunDrag.addParameter(pType_hessian_active)
### END # FOR DRAG OBJECTIVE HESSIAN ###

### FOR GEOMETRY CONSTRAINT ###
geometryFDRun = ExternalRun("GEOMETRY", "mpirun -n 4 SU2_GEO config_Optimization_tmpl.cfg",True)
geometryFDRun.addConfig("config_Optimization_tmpl.cfg")
geometryFDRun.addData("DEFORM/mesh_tutorial_ffd_deform.su2")
geometryFDRun.addParameter(pType_FDstep)
geometryFDRun.addParameter(pType_direct)
geometryFDRun.addParameter(pType_Iter_run)
geometryFDRun.addParameter(pType_mesh_filename_deformed)
geometryFDRun.addParameter(pType_ObjFun_DRAG)
geometryFDRun.addParameter(pType_hessian_passive)
### END # FOR GEOMETRY CONSTRAINT ###

### Define Function and Constraints out of the runs ###
fun = Function("DRAG","DIRECT/history.dat",TableReader(0,0,start=(-1,10),end=(None,None),delim=","))
fun.addInputVariable(var,"DOT_DRAG/of_grad.dat",TableReader(None,0,start=(1,0),end=(None,None)))
fun.addValueEvalStep(meshDeformationRun)
fun.addValueEvalStep(directRun)
fun.addGradientEvalStep(adjointRunDrag)
fun.addGradientEvalStep(dotProductRunDrag)
fun.addGradientEvalStep(hessianRunDrag)

thicknessConstraint1 = Function("GEOMETRY","GEOMETRY/of_func.dat",TableReader(0,0,start=(-1,18),end=(None,None),delim=","))
thicknessConstraint1.addInputVariable(var,"GEOMETRY/of_grad.dat",TableReader(None,0,start=(4,19),end=(None,None),delim=","))
thicknessConstraint1.addValueEvalStep(meshDeformationRun)
thicknessConstraint1.addValueEvalStep(geometryFDRun)
thicknessConstraint1.addGradientEvalStep(geometryFDRun)

thicknessConstraint2 = Function("GEOMETRY","GEOMETRY/of_func.dat",TableReader(0,0,start=(-1,19),end=(None,None),delim=","))
thicknessConstraint2.addInputVariable(var,"GEOMETRY/of_grad.dat",TableReader(None,0,start=(4,20),end=(None,None),delim=","))
thicknessConstraint2.addValueEvalStep(meshDeformationRun)
thicknessConstraint2.addValueEvalStep(geometryFDRun)
thicknessConstraint2.addGradientEvalStep(geometryFDRun)

thicknessConstraint3 = Function("GEOMETRY","GEOMETRY/of_func.dat",TableReader(0,0,start=(-1,20),end=(None,None),delim=","))
thicknessConstraint3.addInputVariable(var,"GEOMETRY/of_grad.dat",TableReader(None,0,start=(4,21),end=(None,None),delim=","))
thicknessConstraint3.addValueEvalStep(meshDeformationRun)
thicknessConstraint3.addValueEvalStep(geometryFDRun)
thicknessConstraint3.addGradientEvalStep(geometryFDRun)

thicknessConstraint4 = Function("GEOMETRY","GEOMETRY/of_func.dat",TableReader(0,0,start=(-1,21),end=(None,None),delim=","))
thicknessConstraint4.addInputVariable(var,"GEOMETRY/of_grad.dat",TableReader(None,0,start=(4,22),end=(None,None),delim=","))
thicknessConstraint4.addValueEvalStep(meshDeformationRun)
thicknessConstraint4.addValueEvalStep(geometryFDRun)
thicknessConstraint4.addGradientEvalStep(geometryFDRun)

thicknessConstraint5 = Function("GEOMETRY","GEOMETRY/of_func.dat",TableReader(0,0,start=(-1,22),end=(None,None),delim=","))
thicknessConstraint5.addInputVariable(var,"GEOMETRY/of_grad.dat",TableReader(None,0,start=(4,23),end=(None,None),delim=","))
thicknessConstraint5.addValueEvalStep(meshDeformationRun)
thicknessConstraint5.addValueEvalStep(geometryFDRun)
thicknessConstraint5.addGradientEvalStep(geometryFDRun)
### END # Define Function and Constraints out of the runs ###

# Driver
driver = ScipyDriver()

def_objs = config['OPT_OBJECTIVE']
this_obj = def_objs.keys()[0]
sign  = SU2.io.get_objectiveSign(this_obj)
driver.addObjective("min", fun, sign)

driver.addLowerBound(thicknessConstraint1, 0.077, 1.0)
driver.addLowerBound(thicknessConstraint2, 0.072, 1.0)
driver.addLowerBound(thicknessConstraint3, 0.066, 1.0)
driver.addLowerBound(thicknessConstraint4, 0.060, 1.0)
driver.addLowerBound(thicknessConstraint5, 0.054, 1.0)

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
                         xb=xbounds,
                         driver=driver)

log.close()
his.close()
  
print ('Finished')




