from FADO import *
import SU2
import scipy.optimize
import subprocess
import numpy as np
from scipy.optimize import minimize
from externalrunextension import *

config = SU2.io.Config("RAE2822_optimization_original.cfg")
designparams = copy.deepcopy(config['DV_VALUE_OLD'])

var = InputVariable(np.array(designparams),ArrayLabelReplacer("__X__"))
