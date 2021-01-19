from FADO import *
import SU2
import scipy.optimize
import subprocess
import numpy as np
from scipy.optimize import minimize
from externalrunextension import *


config = SU2.io.Config("flatback_onemarker_original.cfg")
designparams = copy.deepcopy(config['DV_VALUE_OLD'])

var = InputVariable(np.array(designparams),ArrayLabelReplacer("__X__"))

pType_direct = Parameter(["DIRECT"],LabelReplacer("__MATH_PROBLEM__"))
pType_adjoint = Parameter(["DISCRETE_ADJOINT"],LabelReplacer("__MATH_PROBLEM__"))
pType_mesh_filename_original = Parameter(["cutoff_onemarker.su2"],LabelReplacer("__MESH_FILENAME__"))
pType_mesh_filename_deformed = Parameter(["cutoff_onemarker_def.su2"],LabelReplacer("__MESH_FILENAME__"))

pType_ObjFun_DRAG = Parameter(["DRAG"],LabelReplacer("__OBJECTIVE_FUNCTION__"))
pType_ObjFun_LIFT = Parameter(["LIFT"],LabelReplacer("__OBJECTIVE_FUNCTION__"))

meshDeformationRun = SU2MeshDeformationSkipFirstIteration("DEFORM","mpirun -n 4 SU2_DEF config_tmpl.cfg",True,"config_tmpl.cfg")
meshDeformationRun.addConfig("config_tmpl.cfg")
meshDeformationRun.addData("cutoff_onemarker.su2")
meshDeformationRun.addParameter(pType_direct)
meshDeformationRun.addParameter(pType_mesh_filename_original)
meshDeformationRun.addParameter(pType_ObjFun_DRAG) #not actually needed, but used to make a valid config file
 
directRun = ExternalSU2CFDSingleZoneDriverWithRestartOption("DIRECT","mpirun -n 4 SU2_CFD config_tmpl.cfg",True,"config_tmpl.cfg")
directRun.addConfig("config_tmpl.cfg")
directRun.addData("DEFORM/cutoff_onemarker_def.su2")
directRun.addData("solution_flow.dat") #dummy solution file
directRun.addParameter(pType_direct)
directRun.addParameter(pType_mesh_filename_deformed)
directRun.addParameter(pType_ObjFun_DRAG)

adjointRunDrag = ExternalSU2CFDDiscAdjSingleZoneDriverWithRestartOption("ADJOINT_DRAG","mpirun -n 4 SU2_CFD_AD config_tmpl.cfg",True,"config_tmpl.cfg")
adjointRunDrag.addConfig("config_tmpl.cfg")
adjointRunDrag.addData("DEFORM/cutoff_onemarker_def.su2")
adjointRunDrag.addData("DIRECT/solution_flow.dat")
adjointRunDrag.addData("solution_adj_cd.dat") #dummy adj soluion file
adjointRunDrag.addParameter(pType_adjoint)
adjointRunDrag.addParameter(pType_mesh_filename_deformed)
adjointRunDrag.addParameter(pType_ObjFun_DRAG)

dotProductRunDrag = ExternalRun("DOT_DRAG","mpirun -n 4 SU2_DOT_AD config_tmpl.cfg",True)
dotProductRunDrag.addConfig("config_tmpl.cfg")
dotProductRunDrag.addData("DEFORM/cutoff_onemarker_def.su2")
dotProductRunDrag.addData("ADJOINT_DRAG/solution_adj_cd.dat")
dotProductRunDrag.addParameter(pType_adjoint)
dotProductRunDrag.addParameter(pType_mesh_filename_deformed)
dotProductRunDrag.addParameter(pType_ObjFun_DRAG)

### FOR LIFT CONSTRAINT ###
adjointRunLift = ExternalSU2CFDDiscAdjSingleZoneDriverWithRestartOption("ADJOINT_LIFT","mpirun -n 4 SU2_CFD_AD config_tmpl.cfg",True,"config_tmpl.cfg")
adjointRunLift.addConfig("config_tmpl.cfg")
adjointRunLift.addData("DEFORM/cutoff_onemarker_def.su2")
adjointRunLift.addData("DIRECT/solution_flow.dat")
adjointRunLift.addData("solution_adj_cl.dat") #dummy adj soluion file
adjointRunLift.addParameter(pType_adjoint)
adjointRunLift.addParameter(pType_mesh_filename_deformed)
adjointRunLift.addParameter(pType_ObjFun_LIFT)

dotProductRunLift = ExternalRun("DOT_LIFT","mpirun -n 4 SU2_DOT_AD config_tmpl.cfg",True)
dotProductRunLift.addConfig("config_tmpl.cfg")
dotProductRunLift.addData("DEFORM/cutoff_onemarker_def.su2")
dotProductRunLift.addData("ADJOINT_LIFT/solution_adj_cl.dat")
dotProductRunLift.addParameter(pType_adjoint)
dotProductRunLift.addParameter(pType_mesh_filename_deformed)
dotProductRunLift.addParameter(pType_ObjFun_LIFT)
### END # FOR LIFT CONSTRAINT ###

fun = Function("DRAG","DIRECT/convhist.csv",TableReader(0,0,start=(-1,8),end=(None,None),delim=","))
fun.addInputVariable(var,"DOT_DRAG/of_grad.dat",TableReader(None,0,start=(1,0),end=(None,None)))
#fun.addPreProcessStep(preDirectRun)
fun.addValueEvalStep(meshDeformationRun)
fun.addValueEvalStep(directRun)
fun.addGradientEvalStep(adjointRunDrag)
fun.addGradientEvalStep(dotProductRunDrag)

liftConstraint = Function("LIFT","DIRECT/convhist.csv",TableReader(0,0,start=(-1,9),end=(None,None),delim=","))
liftConstraint.addInputVariable(var,"DOT_LIFT/of_grad.dat",TableReader(None,0,start=(1,0),end=(None,None)))
liftConstraint.addValueEvalStep(meshDeformationRun)
liftConstraint.addValueEvalStep(directRun)
liftConstraint.addGradientEvalStep(adjointRunLift)
liftConstraint.addGradientEvalStep(dotProductRunLift)

# Driver
driver = ScipyDriver()

def_objs = config['OPT_OBJECTIVE']
this_obj = def_objs.keys()[0]
scale = def_objs[this_obj]['SCALE']
global_factor = float(config['OPT_GRADIENT_FACTOR'])
sign  = SU2.io.get_objectiveSign(this_obj)
driver.addObjective("min", fun, sign * scale * global_factor)
driver.addLowerBound(liftConstraint, 1.2, 1.0)

directSolutionFilename = "DIRECT/solution_flow.dat"
pathForDirectSolutionFilename = os.path.join(driver._workDir,directSolutionFilename)
commandDirectSolution = "cp" + " " + pathForDirectSolutionFilename + " ."
print("command 1: ", commandDirectSolution) 

#driver.addDataFileToFetchAfterValueEval("DIRECT/solution_flow.dat")
driver.setUserPostProcessFun(commandDirectSolution)


adjointSolutionDRAG = "ADJOINT_DRAG/solution_adj_cd.dat"
pathForAdjointSolutionDRAG = os.path.join(driver._workDir,adjointSolutionDRAG)
commandAdjointSolutionDRAG = "cp" + " " + pathForAdjointSolutionDRAG + " ."
print("command 2, part_a: ", commandAdjointSolutionDRAG)

adjointSolutionLIFT = "ADJOINT_LIFT/solution_adj_cl.dat"
pathForAdjointSolutionLIFT = os.path.join(driver._workDir,adjointSolutionLIFT)
commandAdjointSolutionLIFT = "cp" + " " + pathForAdjointSolutionLIFT + " ."
print("command 2, part_b: ", commandAdjointSolutionLIFT)

commandAdjointSolution = commandAdjointSolutionDRAG + " && " + commandAdjointSolutionLIFT
print("command 2, FULL: ", commandAdjointSolution)

#driver.addDataFileToFetchAfterGradientEval("ADJOINT/solution_adj_cd.dat")
driver.setUserPostProcessGrad(commandAdjointSolution)

#        for file in self._dataFilesToRetrieveAfterValueEval:
#            source = os.path.join(self._workDir,file) #source = os.path.join(self._workDir,os.path.basename(file))
#            target = os.path.basename(file)#target = os.path.join(self._userDir,os.path.basename(file))
            #shutil.copy()#(shutil.copy,os.symlink)[self._symLinks](os.path.abspath(file),target)
            #target = os.path.join(self._workDir,os.path.basename(file))
#            shutil.copy(source,target)

driver.preprocess()
driver.setEvaluationMode(False)
driver.setStorageMode(True)

log = open("log.txt","w",1)
his = open("history.txt","w",1)
driver.setLogger(log)
driver.setHistorian(his)

x  = driver.getInitial()

maxIter      = int (config.OPT_ITERATIONS)                      # number of opt iterations
bound_upper       = float (config.OPT_BOUND_UPPER)                   # variable bound to be scaled by the line search
bound_lower       = float (config.OPT_BOUND_LOWER)                   # variable bound to be scaled by the line search
relax_factor      = float (config.OPT_RELAX_FACTOR)                  # line search scale
gradient_factor   = float (config.OPT_GRADIENT_FACTOR)               # objective function and gradient scale

accu = float (config.OPT_ACCURACY) * gradient_factor            # optimizer accuracy

xb_low = [float(bound_lower)/float(relax_factor)]*driver._nVar      # lower dv bound it includes the line search acceleration factor
xb_up  = [float(bound_upper)/float(relax_factor)]*driver._nVar      # upper dv bound it includes the line search acceleration fa
xbounds = list(zip(xb_low, xb_up)) # design bounds

# scale accuracy
eps = 1.0e-04

funcVal = driver.fun(np.array(designparams))
grad = driver.grad(np.array(designparams))

outputs = fmin_slsqp( x0             = x       ,
                      func           = driver.fun   ,
                        f_eqcons       = None               ,
                        f_ieqcons      = None               ,
                      fprime         = driver.grad  ,
                        fprime_eqcons  = None               ,
                        fprime_ieqcons = None               ,
                        args           = None               ,
                      bounds         = xbounds  ,
                      iter           = maxIter  ,
                      iprint         = 2                  ,
                      full_output    = True               ,
                      acc            = accu     ,
                      epsilon        = eps      )

#driver.update()

#driver.setConstraintGradientEvalMode(False)

#outputs = minimize(fun=driver.fun,
#                   x0=x,
#                   jac=driver.grad,
#                   method='SLSQP',
#                   options={'maxiter': maxIter, 'ftol': accu, 'iprint': 2, 'disp': True, 'eps': eps},
#                   tol=accu,
#                   constraints=driver.getConstraints(),
#                   bounds = xbounds)




log.close()
his.close()
  
print ('Finished')




