from FADO import *
import SU2
import scipy.optimize
import subprocess
import numpy as np
from scipy.optimize import minimize
from externalrunextension import *
from ipoptimizer import *

### DEFINE CONFIG, MESH, CONFIG LABELS #
config = SU2.io.Config("optimization_RAE2822_orig.cfg")
designparams = copy.deepcopy(config['DV_VALUE_OLD'])
var = InputVariable(np.array(designparams),ArrayLabelReplacer("__X__"))

pType_FDstep = Parameter(["0.001"+",0.001"*(len(designparams)-1)], LabelReplacer("__X__"))

pType_direct = Parameter(["DIRECT"],LabelReplacer("__MATH_PROBLEM__"))
pType_adjoint = Parameter(["DISCRETE_ADJOINT"],LabelReplacer("__MATH_PROBLEM__"))
pType_mesh_filename_original = Parameter(["rae2822_ffd.su2"],LabelReplacer("__MESH_FILENAME__"))
pType_mesh_filename_deformed = Parameter(["rae2822_ffd_def.su2"],LabelReplacer("__MESH_FILENAME__"))

pType_Iter_run = Parameter(["10"],LabelReplacer("__NUM_ITER__"))
pType_Iter_step = Parameter(["1"],LabelReplacer("__NUM_ITER__"))
pType_hessian_active = Parameter(["YES"],LabelReplacer("__ACTIVATE_HESSIAN__"))
pType_hessian_passive = Parameter(["NO"],LabelReplacer("__ACTIVATE_HESSIAN__"))
pType_comphessian = Parameter(["PARAM_LEVEL_COMPLETE"],LabelReplacer("__SMOOTHING_MODE__"))
pType_gradonly = Parameter(["ONLY_GRADIENT"],LabelReplacer("__SMOOTHING_MODE__"))

pType_ObjFun_DRAG = Parameter(["DRAG"],LabelReplacer("__OBJECTIVE_FUNCTION__"))
pType_ObjFun_LIFT = Parameter(["LIFT"],LabelReplacer("__OBJECTIVE_FUNCTION__"))
pType_ObjFun_MOMENT_Z = Parameter(["MOMENT_Z"],LabelReplacer("__OBJECTIVE_FUNCTION__"))

pType_OneShot_passive = Parameter(["NONE"], LabelReplacer("__ONESHOT_MODE__"))
pType_OneShot_active = Parameter(["PIGGYBACK"], LabelReplacer("__ONESHOT_MODE__"))
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
meshDeformationRun.addParameter(pType_comphessian)
meshDeformationRun.addParameter(pType_OneShot_passive)
### END # FOR MESH DEFORMATION ###

### FOR FLOW AND ADJOINT DRAG OBJECTIVE FUNCTION ###
combinedRun = ExternalSU2CFDOneShotSingleZoneDriverWithRestartOption("COMBINED","mpirun -n 4 SU2_CFD_AD optimization_RAE2822_tmpl.cfg",True,"optimization_RAE2822_tmpl.cfg")
combinedRun.addConfig("optimization_RAE2822_tmpl.cfg")
combinedRun.addData("DEFORM/rae2822_ffd_def.su2")
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
combinedRunLift = ExternalSU2CFDOneShotSingleZoneDriverWithRestartOption("COMBINED_LIFT","mpirun -n 4 SU2_CFD_AD optimization_RAE2822_tmpl.cfg",True,"optimization_RAE2822_tmpl.cfg")
combinedRunLift.addConfig("optimization_RAE2822_tmpl.cfg")
combinedRunLift.addData("DEFORM/rae2822_ffd_def.su2")
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

### FOR FLOW AND ADJOINT MOMENTUM CONSTRAINT ###
combinedRunMomZ = ExternalSU2CFDOneShotSingleZoneDriverWithRestartOption("COMBINED_MOMENT_Z","mpirun -n 4 SU2_CFD_AD optimization_RAE2822_tmpl.cfg",True,"optimization_RAE2822_tmpl.cfg")
combinedRunMomZ.addConfig("optimization_RAE2822_tmpl.cfg")
combinedRunMomZ.addData("DEFORM/rae2822_ffd_def.su2")
combinedRunMomZ.addData("solution_flow.dat") #has to be an actual solution file
combinedRunMomZ.addData("solution_adj_cmz.dat") #restart adj solution file
combinedRunMomZ.addParameter(pType_adjoint)
combinedRunMomZ.addParameter(pType_Iter_run)
combinedRunMomZ.addParameter(pType_mesh_filename_deformed)
combinedRunMomZ.addParameter(pType_ObjFun_MOMENT_Z)
combinedRunMomZ.addParameter(pType_hessian_active)
combinedRunMomZ.addParameter(pType_gradonly)
combinedRunMomZ.addParameter(pType_OneShot_active)
### END # FOR FLOW AND ADJOINT MOMENTUM CONSTRAINT ###

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
geometryFDRun.addParameter(pType_comphessian)
geometryFDRun.addParameter(pType_OneShot_passive)
### END # FOR GEOMETRY CONSTRAINT ###

### Define Function and Constraints out of the runs ###
fun = Function("DRAG","COMBINED/history.csv",TableReader(0,0,start=(-1,14),end=(None,None),delim=","))
fun.addInputVariable(var,"COMBINED/orig_grad.dat",TableReader(0,None,start=(0,0),end=(None,None),delim=","))
fun.addValueEvalStep(meshDeformationRun)
fun.addValueEvalStep(combinedRun)

liftConstraint = Function("LIFT","COMBINED/history.csv",TableReader(0,0,start=(-1,15),end=(None,None),delim=","))
liftConstraint.addInputVariable(var,"COMBINED_LIFT/orig_grad.dat",TableReader(0,None,start=(0,0),end=(None,None),delim=","))
liftConstraint.addValueEvalStep(meshDeformationRun)
liftConstraint.addGradientEvalStep(combinedRunLift)

momentConstraint = Function("MOMENT_Z","COMBINED/history.csv",TableReader(0,0,start=(-1,19),end=(None,None),delim=","))
momentConstraint.addInputVariable(var,"COMBINED_MOMENT_Z/orig_grad.dat",TableReader(0,None,start=(0,0),end=(None,None),delim=","))
momentConstraint.addValueEvalStep(meshDeformationRun)
momentConstraint.addGradientEvalStep(combinedRunMomZ)

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
directSolutionFilename = "COMBINED/solution_flow.dat"
pathForDirectSolutionFilename = os.path.join(driver._workDir,directSolutionFilename)
commandDirectSolution = "cp" + " " + pathForDirectSolutionFilename + " ."
print("command 1: ", commandDirectSolution) 

adjointSolutionDRAG = "COMBINED/solution_adj_cd.dat"
pathForAdjointSolutionDRAG = os.path.join(driver._workDir,adjointSolutionDRAG)
commandAdjointSolutionDRAG = "cp" + " " + pathForAdjointSolutionDRAG + " ."
print("command 2: ", commandAdjointSolutionDRAG)
driver.setUserPostProcessGrad(commandDirectSolution + "&&" + commandAdjointSolutionDRAG + "&&" + "echo gradientpostprocessing")

adjointSolutionLIFT = "COMBINED_LIFT/solution_adj_cl.dat"
pathForAdjointSolutionLIFT = os.path.join(driver._workDir,adjointSolutionLIFT)
commandAdjointSolutionLIFT = "cp" + " " + pathForAdjointSolutionLIFT + " ."
print("command 3: ", commandAdjointSolutionLIFT)
driver.setUserPostProcessEqConGrad(commandAdjointSolutionLIFT + "&&" + "echo liftpostprocessing")

adjointSolutionMOMZ = "COMBINED_MOMENT_Z/solution_adj_cmz.dat"
pathForAdjointSolutionMOMZ = os.path.join(driver._workDir,adjointSolutionMOMZ)
commandAdjointSolutionMOMZ = "cp" + " " + pathForAdjointSolutionMOMZ + " ."
print("command 4: ", commandAdjointSolutionMOMZ)
driver.setUserPostProcessIEqConGrad(commandAdjointSolutionMOMZ + "&&" + "echo momentpostprocessing")
### END # Postprocess command ###

driver.preprocess()
driver.setEvaluationMode(False)
driver.setStorageMode(True,"DESIGN/DSN_")

log = open("log.txt","w",1)
his = open("history.txt","w",1)
driver.setLogger(log)
driver.setHistorian(his)

x  = driver.getInitial()
driver.setConstraintGradientEvalMode(False)
driver.hessian_eval_parameters("COMBINED", "of_hess.dat")

conf = Ipoptconfig()
conf.nparam = len(x)
conf.neqcons = 1
conf.nieqcons = 2
conf.lb = config.OPT_BOUND_LOWER
conf.ub = config.OPT_BOUND_UPPER
conf.lower = np.array([0.0, 0.0])
conf.upper = np.array([100.0, 100.0])
conf.eps3=0.0

outputs = Ipoptimizer(x0=x,
                      func=driver.fun,
                      f_eqcons= driver.eq_cons,
                      f_ieqcons= driver.ieq_cons,
                      fprime=driver.grad,
                      fprime_eqcons= driver.eq_cons_grad,
                      fprime_ieqcons= driver.ieq_cons_grad,
                      fdotdot= None,
                      config=conf)

log.close()
his.close()
  
print ('Finished')
