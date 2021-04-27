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

pType_Iter_run = Parameter(["10"],LabelReplacer("__NUM_ITER__"))
pType_Iter_step = Parameter(["1"],LabelReplacer("__NUM_ITER__"))
pType_hessian_active = Parameter(["YES"],LabelReplacer("__ACTIVATE_HESSIAN__"))
pType_hessian_passive = Parameter(["NO"],LabelReplacer("__ACTIVATE_HESSIAN__"))
pType_comphessian = Parameter(["PARAM_LEVEL_COMPLETE"],LabelReplacer("__SMOOTHING_MODE__"))
pType_gradonly = Parameter(["ONLY_GRADIENT"],LabelReplacer("__SMOOTHING_MODE__"))

pType_ObjFun_DRAG = Parameter(["DRAG"],LabelReplacer("__OBJECTIVE_FUNCTION__"))
pType_ObjFun_LIFT = Parameter(["LIFT"],LabelReplacer("__OBJECTIVE_FUNCTION__"))

pType_OneShot_passive = Parameter(["NONE"], LabelReplacer("__ONESHOT_MODE__"))
pType_OneShot_active = Parameter(["PIGGYBACK"], LabelReplacer("__ONESHOT_MODE__"))
### END # DEFINE CONFIG, MESH, CONFIG LABELS #


### FOR MESH DEFORMATION ###
meshDeformationRun = SU2MeshDeformationSkipFirstIteration("DEFORM","mpirun -n 4 SU2_DEF config_Optimization_tmpl.cfg",True,"config_Optimization_tmpl.cfg")
meshDeformationRun.addConfig("config_Optimization_tmpl.cfg")
meshDeformationRun.addData("mesh_tutorial_ffd.su2")
meshDeformationRun.addParameter(pType_direct)
meshDeformationRun.addParameter(pType_Iter_run)
meshDeformationRun.addParameter(pType_mesh_filename_original)
meshDeformationRun.addParameter(pType_ObjFun_DRAG) #not actually needed, but used to make a valid config file
meshDeformationRun.addParameter(pType_hessian_passive)
meshDeformationRun.addParameter(pType_gradonly)
meshDeformationRun.addParameter(pType_OneShot_passive)
### END # FOR MESH DEFORMATION ###

### FOR FLOW AND ADJOINT DRAG OBJECTIVE FUNCTION ###
combinedRun = ExternalSU2CFDOneShotSingleZoneDriverWithRestartOption("COMBINED","mpirun -n 4 SU2_CFD_AD config_Optimization_tmpl.cfg",True,"config_Optimization_tmpl.cfg")
combinedRun.addConfig("config_Optimization_tmpl.cfg")
combinedRun.addData("DEFORM/mesh_tutorial_ffd_deform.su2")
combinedRun.addData("solution_flow.dat") #has to be an actual solution file
combinedRun.addData("solution_adj_cd.dat") #restart adj solution file
combinedRun.addParameter(pType_adjoint)
combinedRun.addParameter(pType_Iter_run)
combinedRun.addParameter(pType_mesh_filename_deformed)
combinedRun.addParameter(pType_ObjFun_DRAG)
combinedRun.addParameter(pType_hessian_active)
combinedRun.addParameter(pType_comphessian)
combinedRun.addParameter(pType_OneShot_active)
### END # FOR FLOW AND ADJOINT DRAG OBJECTIVE FUNCTION ###

### FOR FLOW AND ADJOINT LIFT CONSTRAINT ###
combinedRunLift = ExternalSU2CFDOneShotSingleZoneDriverWithRestartOption("COMBINED_LIFT","mpirun -n 4 SU2_CFD_AD config_Optimization_tmpl.cfg",True,"config_Optimization_tmpl.cfg")
combinedRunLift.addConfig("config_Optimization_tmpl.cfg")
combinedRunLift.addData("DEFORM/mesh_tutorial_ffd_deform.su2")
combinedRunLift.addData("solution_flow.dat") #has to be an actual solution file
combinedRunLift.addData("solution_adj_cl.dat") #restart adj solution file
combinedRunLift.addParameter(pType_adjoint)
combinedRunLift.addParameter(pType_Iter_run)
combinedRunLift.addParameter(pType_mesh_filename_deformed)
combinedRunLift.addParameter(pType_ObjFun_LIFT)
combinedRunLift.addParameter(pType_hessian_active)
combinedRunLift.addParameter(pType_gradonly)
combinedRunLift.addParameter(pType_OneShot_active)
### END # FOR FLOW AND ADJOINT LIFT CONSTRAINT ###

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
geometryFDRun.addParameter(pType_comphessian)
geometryFDRun.addParameter(pType_OneShot_passive)
### END # FOR GEOMETRY CONSTRAINT ###

### Define Function and Constraints out of the runs ###
fun = Function("DRAG","COMBINED/history.dat",TableReader(0,0,start=(-1,17),end=(None,None),delim=","))
fun.addInputVariable(var,"COMBINED/orig_grad.dat",TableReader(0,None,start=(0,0),end=(None,None),delim=","))
fun.addValueEvalStep(meshDeformationRun)
fun.addValueEvalStep(combinedRun)

liftConstraint = Function("LIFT","COMBINED_LIFT/history.dat",TableReader(0,0,start=(-1,18),end=(None,None),delim=","))
liftConstraint.addInputVariable(var,"COMBINED_LIFT/orig_grad.dat",TableReader(0,None,start=(0,0),end=(None,None),delim=","))
liftConstraint.addValueEvalStep(meshDeformationRun)
liftConstraint.addValueEvalStep(combinedRunLift)

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

driver.addEquality(liftConstraint, 0.2514, 1.0)
driver.addLowerBound(thicknessConstraint1, 0.077, 1.0)
driver.addLowerBound(thicknessConstraint2, 0.072, 1.0)
driver.addLowerBound(thicknessConstraint3, 0.066, 1.0)
driver.addLowerBound(thicknessConstraint4, 0.060, 1.0)
driver.addLowerBound(thicknessConstraint5, 0.054, 1.0)

### Postprocess command ###
directSolutionFilename = "COMBINED/solution_flow.dat"
pathForDirectSolutionFilename = os.path.join(driver._workDir,directSolutionFilename)
commandDirectSolution = "cp" + " " + pathForDirectSolutionFilename + " ."
print("command 1: ", commandDirectSolution) 

adjointSolutionDRAG = "COMBINED/solution_adj_cd.dat"
pathForAdjointSolutionDRAG = os.path.join(driver._workDir,adjointSolutionDRAG)
commandAdjointSolutionDRAG = "cp" + " " + pathForAdjointSolutionDRAG + " ."
print("command 2: ", commandAdjointSolutionDRAG)
driver.setUserPostProcessGrad(commandDirectSolution + " && " + commandAdjointSolutionDRAG)

adjointSolutionLIFT = "COMBINED_LIFT/solution_adj_cl.dat"
pathForAdjointSolutionLIFT = os.path.join(driver._workDir,adjointSolutionLIFT)
commandAdjointSolutionLIFT = "cp" + " " + pathForAdjointSolutionLIFT + " ."
print("command 3: ", commandAdjointSolutionLIFT)
driver.setUserPostProcessEqConGrad(commandAdjointSolutionLIFT)
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

driver.hessian_eval_parameters("COMBINED", "of_hess.dat")

conf = RSQPconfig()
conf.hybrid_sobolev=True
conf.bfgs = optimize.BFGS(exception_strategy='damp_update', init_scale=1.0)
conf.bfgs.initialize(len(x),'hess')

outputs = SQPconstrained(x0=x,
                         func=driver.fun,
                         f_eqcons= driver.eq_cons,
                         f_ieqcons= [],
                         fprime=driver.grad,
                         fprime_eqcons= driver.eq_cons_grad,
                         fprime_ieqcons= [],
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




