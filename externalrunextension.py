import os
import shutil

import SU2

from FADO.evaluation import ExternalRun

class ExternalRunWithPreAndPostProcess(ExternalRun):
  """
  A child class that adds preprocess and postprocess routines to allow additional manipulations that are based on a config file
  """
  def __init__(self, dir, command, useSymLinks=False):
    ExternalRun.__init__(self, dir, command, useSymLinks)
  #end
  
  def preProcess(self):
    pass
  #end
  
  def postProcess(self):
    pass
  #end
  
  def run(self,timeout=None):
    """Start the process and wait for it to finish."""
    if not self._isIni:
      raise RuntimeError("Run was not initialized.")
    if self._numTries == self._maxTries:
      raise RuntimeError("Run failed.")
    if self._isRun:
      return self._retcode
    
    preExecutionDir = os.getcwd()
    #preprocess
    #change to _workDir
    os.chdir(self._workDir)
    
    self.preProcess()
    
    self._retcode = self._process.wait(timeout)
    
    self._numTries += 1
    
    if not self._success():
      self.finalize()
      self._createProcess()
      self._isIni = True
      return self.run(timeout)
    #end
    
    #postprocess
    self.postProcess()
    
    #after finish change to _userDir
    os.chdir(preExecutionDir)  
    
    self._numTries = 0
    self._isRun = True
    return self._retcode
  #end
  
class ExternalSU2CFDSingleZoneDriverWithRestartOption(ExternalRunWithPreAndPostProcess):
    def __init__(self, dir, command, useSymLinks=False, mainConfigName="config_tmpl.cfg"):
        ExternalRunWithPreAndPostProcess.__init__(self, dir, command, useSymLinks)
        self._mainConfigName = mainConfigName
        self._numberOfSuccessfulRuns = 0 #this variable will be used to indicate whether the restart option in config should be YES or NO
    #end    
    
    def preProcess(self):
        #by default RESTART_SOL=NO as on the initial step we don't have prior information
        #after the first iteration, this should be changed to YES
        if self._numberOfSuccessfulRuns > 0:
              #we are already inside the DIRECT folder, fetch the config and change the RESTART_SOL parameter
              config = SU2.io.Config(self._mainConfigName)
              config['RESTART_SOL'] = 'YES'
              config.dump(self._mainConfigName)
    #end
    
    def postProcess(self):
        self._numberOfSuccessfulRuns += 1
        #here one has to rename restart file to solution file that the adjoint solver can use
        #RESTART TO SOLUTION
        config = SU2.io.Config(self._mainConfigName)
        
        restart  = config.RESTART_FILENAME
        solution = config.SOLUTION_FILENAME
        #print("Restart file name: ", restart)
        #print("Current directory: ", os.getcwd())
        if os.path.exists(restart):
            shutil.move( restart , solution )
    #end
    def enableRestart(self):
        if self._numberOfSuccessfulRuns == 0:
            self._numberOfSuccessfulRuns += 1
            print("RESTART OPTION WILL BE ENABLED ON EXECUTION OF SU2_CFD")
    #end
#end

class ExternalSU2CFDDiscAdjSingleZoneDriverWithRestartOption(ExternalRunWithPreAndPostProcess):
    def __init__(self, dir, command, useSymLinks=False, mainConfigName="config_tmpl.cfg"):
        ExternalRunWithPreAndPostProcess.__init__(self, dir, command, useSymLinks)
        self._mainConfigName = mainConfigName
        self._numberOfSuccessfulRuns = 0 #this variable will be used to indicate whether the restart option in config should be YES or NO
    #end
    
    def preProcess(self):
        if self._numberOfSuccessfulRuns > 0:
              #we are already inside the ADJOINT folder, fetch the config and change the RESTART_SOL parameter
              config = SU2.io.Config(self._mainConfigName)
              config['RESTART_SOL'] = 'YES'
              config.dump(self._mainConfigName)
    #end
          
    def postProcess(self):
        self._numberOfSuccessfulRuns += 1
        #RESTART TO SOLUTION
        config = SU2.io.Config(self._mainConfigName)
        restart  = config.RESTART_ADJ_FILENAME
        solution = config.SOLUTION_ADJ_FILENAME
        # add suffix
        func_name = config.OBJECTIVE_FUNCTION
        suffix    = SU2.io.get_adjointSuffix(func_name)
        restart   = SU2.io.add_suffix(restart,suffix)
        solution  = SU2.io.add_suffix(solution,suffix)
        
        if os.path.exists(restart):
            shutil.move( restart , solution )
    #end
    
    def enableRestart(self):
        if self._numberOfSuccessfulRuns == 0:
            self._numberOfSuccessfulRuns += 1
            print("RESTART OPTION WILL BE ENABLED ON EXECUTION OF SU2_CFD_AD")
    #end
#end

class SU2MeshDeformationSkipFirstIteration(ExternalRunWithPreAndPostProcess):
    def __init__(self, dir, command, useSymLinks=False, mainConfigName="config_tmpl.cfg"):
        ExternalRunWithPreAndPostProcess.__init__(self, dir, command, useSymLinks)
        self._mainConfigName = mainConfigName
        self._isAlreadyCalledForTheFirstTime = False
    #end
    
    def initialize(self):
        """
        Initialize the run, create the subdirectory, copy/symlink the data and
        configuration files, and write the parameters and variables to the latter.
        """
        if self._isIni: return

        os.mkdir(self._workDir)
        for file in self._dataFiles:
            target = os.path.join(self._workDir,os.path.basename(file))
            (shutil.copy,os.symlink)[self._symLinks](os.path.abspath(file),target)

        for file in self._confFiles:
            target = os.path.join(self._workDir,os.path.basename(file))
            shutil.copy(file,target)
            for par in self._parameters:
                par.writeToFile(target)
            for var in self._variables:
                var.writeToFile(target)

        if self._isAlreadyCalledForTheFirstTime:
            self._createProcess()
        self._isIni = True
        self._isRun = False
        self._numTries = 0
    #end
        
    def run(self,timeout=None):
        #print("MESH DEFORMATION EVALUATED")
        if self._isAlreadyCalledForTheFirstTime: 
            ExternalRunWithPreAndPostProcess.run(self,timeout)
        else:
            currentDir = os.getcwd()
            #print("CURRENT DIR ", currentDir)
            #change to workDir
            os.chdir(self._workDir)
            #print("CURRENT DIR ", os.getcwd())
            config = SU2.io.Config(self._mainConfigName)
            mesh_name = config['MESH_FILENAME']
            mesh_out_name = config['MESH_OUT_FILENAME']
            if os.path.exists(mesh_name):
                shutil.copy( mesh_name , mesh_out_name )
                
            self._isAlreadyCalledForTheFirstTime = True
            #back to previous dir
            os.chdir(currentDir)
            
            self._numTries = 0
            self._isRun = True
            return 1
    #end
#end
