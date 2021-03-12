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

pType_mesh_filename_original = Parameter(["mesh_NACA0012_inv.su2"],LabelReplacer("__MESH_FILENAME__"))

pType_hessian_active = Parameter(["YES"],LabelReplacer("__ACTIVATE_HESSIAN__"))
pType_ObjFun_DRAG = Parameter(["DRAG"],LabelReplacer("__OBJECTIVE_FUNCTION__"))
pType_Iter_OneShot_iter = Parameter(["100"],LabelReplacer("__OSITER__"))
pType_Iter_piggysteps = Parameter(["100"],LabelReplacer("__NUM_ITER__"))
pType_OneShot_active = Parameter(["ONESHOT"], LabelReplacer("__ONESHOT_MODE__"))
pType_OneShot_step = Parameter(["1e-3"], LabelReplacer("__OSSTEP__"))

### ALL IN ONE RUN ###
optimizerRun = ExternalSU2CFDOneShotSingleZoneDriverWithRestartOption("DIRECT","mpirun -n 4 SU2_CFD_AD naca0012_config_tmpl.cfg",True,"naca0012_config_tmpl.cfg")
optimizerRun.addConfig("naca0012_config_tmpl.cfg")
optimizerRun.addData("mesh_NACA0012_inv.su2")
optimizerRun.addData("solution_flow.dat") #has to be an actual solution file
optimizerRun.addData("solution_adj_cd.dat") #dummy adj solution file
optimizerRun.addParameter(pType_Iter_piggysteps)
optimizerRun.addParameter(pType_mesh_filename_original)
optimizerRun.addParameter(pType_ObjFun_DRAG)
optimizerRun.addParameter(pType_hessian_active)
optimizerRun.addParameter(pType_OneShot_active)
optimizerRun.addParameter(pType_Iter_OneShot_iter)
optimizerRun.addParameter(pType_OneShot_step)
### END # ALL IN ONE RUN ###

### Define Function and Constraints out of the runs ###
fun = Function("RUN","DIRECT/history.csv",TableReader(0,0,start=(-1,12),end=(None,None),delim=","))
fun.addValueEvalStep(optimizerRun)
### END # Define Function and Constraints out of the runs ###

# Driver
driver = ScipyDriver()

def_objs = config['OPT_OBJECTIVE']
this_obj = def_objs.keys()[0]
sign  = SU2.io.get_objectiveSign(this_obj)
driver.addObjective("min", fun, sign)

### Postprocess command ###
directSolutionFilename = "RUN/solution_flow.dat"
pathForDirectSolutionFilename = os.path.join(driver._workDir,directSolutionFilename)
commandDirectSolution = "cp" + " " + pathForDirectSolutionFilename + " ."
print("command 1: ", commandDirectSolution)

adjointSolutionDRAG = "RUN/solution_adj_cd.dat"
pathForAdjointSolutionDRAG = os.path.join(driver._workDir,adjointSolutionDRAG)
commandAdjointSolutionDRAG = "cp" + " " + pathForAdjointSolutionDRAG + " ."
print("command 2: ", commandAdjointSolutionDRAG)
driver.setUserPostProcessGrad(commandDirectSolution + "&&" + commandAdjointSolutionDRAG)
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

driver.fun(x)

log.close()
his.close()
  
print ('Finished')
