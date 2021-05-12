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

pType_mesh_filename_original = Parameter(["rae2822_ffd.su2"],LabelReplacer("__MESH_FILENAME__"))
pType_ObjFun_DRAG = Parameter(["DRAG"],LabelReplacer("__OBJECTIVE_FUNCTION__"))

pType_Iter_piggysteps = Parameter(["100"],LabelReplacer("__NUM_ITER__"))
pType_hessian_active = Parameter(["YES"],LabelReplacer("__ACTIVATE_HESSIAN__"))

pType_OneShot_active = Parameter(["ONESHOT"], LabelReplacer("__ONESHOT_MODE__"))
pType_Iter_OneShot_iter = Parameter(["10"],LabelReplacer("__OSITER__"))
pType_OneShot_step = Parameter(["1e2"], LabelReplacer("__OSSTEP__"))
### END # DEFINE CONFIG, MESH, CONFIG LABELS #

### ALL IN ONE RUN ###
optimizerRun = ExternalSU2CFDOneShotSingleZoneDriverWithRestartOption("ONESHOT","mpirun -n 4 SU2_CFD_AD optimization_RAE2822_tmpl.cfg",True,"optimization_RAE2822_tmpl.cfg")
optimizerRun.addConfig("optimization_RAE2822_tmpl.cfg")
optimizerRun.addData("rae2822_ffd.su2")
optimizerRun.addData("solution_flow.dat") #has to be an actual solution file
optimizerRun.addData("solution_adj_cd.dat") #restart adj solution file
optimizerRun.addData("boundary.dat") #necessary for mesh output
optimizerRun.addParameter(pType_mesh_filename_original)
optimizerRun.addParameter(pType_ObjFun_DRAG)
optimizerRun.addParameter(pType_Iter_piggysteps)
optimizerRun.addParameter(pType_hessian_active)
optimizerRun.addParameter(pType_OneShot_active)
optimizerRun.addParameter(pType_Iter_OneShot_iter)
optimizerRun.addParameter(pType_OneShot_step)
### END # ALL IN ONE RUN ###

### Define Function and Constraints out of the runs ###
fun = Function("RUN","ONESHOT/history.csv",TableReader(0,0,start=(-1,14),end=(None,None),delim=","))
fun.addInputVariable(var,"ONESHOT/orig_grad.dat",TableReader(0,None,start=(0,0),end=(None,None),delim=","))
fun.addValueEvalStep(optimizerRun)
### END # Define Function and Constraints out of the runs ###

# Driver
driver = ScipyDriver()

def_objs = config['OPT_OBJECTIVE']
this_obj = def_objs.keys()[0]
sign  = SU2.io.get_objectiveSign(this_obj)
driver.addObjective("min", fun, sign)

### Postprocess command ###
directSolutionFilename = "ONESHOT/solution_flow.dat"
pathForDirectSolutionFilename = os.path.join(driver._workDir,directSolutionFilename)
commandDirectSolution = "cp" + " " + pathForDirectSolutionFilename + " ."
print("command 1: ", commandDirectSolution) 

adjointSolutionDRAG = "ONESHOT/solution_adj_cd.dat"
pathForAdjointSolutionDRAG = os.path.join(driver._workDir,adjointSolutionDRAG)
commandAdjointSolutionDRAG = "cp" + " " + pathForAdjointSolutionDRAG + " ."
print("command 2: ", commandAdjointSolutionDRAG)
driver.setUserPostProcessGrad(commandDirectSolution + "&&" + commandAdjointSolutionDRAG + "&&" + "echo gradientpostprocessing")
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

driver.fun(x)

log.close()
his.close()
  
print ('Finished')
