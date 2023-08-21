import cv2 as cv
import numpy as np
from ProstateLesionDetectionUtils import DatasetsProcessing

class PreprocessProstateDatasets:
    def __init__(self, Data_Dict, Labels_Dict, ResSize = 256):
        """
        PreProcessing Datasets
        KeepInformative() -> PatientProc() -> MaskProc
        Args:
            Data_Dict (dict): data dictionary
            Labels_Dict (dict): labels Dictionary
        """
        self.data = Data_Dict
        self.labels= Labels_Dict
        self.ResSize = ResSize
    
    def KeepInformative(self,IsPicai = True):
        """Throws out patients without lesion"""
        if IsPicai:
            ListKeysToKeep = [key for key in list(self.labels.keys()) if np.unique(self.labels[key]).shape[0] == 2]
            LabDict = {key:self.labels[key] for key in ListKeysToKeep}
            self.labels = LabDict
        else:
            ListKeysToKeep = [key for key in list(self.data.keys()) if key in list(self.labels.keys())]
        PatInst = {key:self.data[key] for key in ListKeysToKeep}
        self.data = PatInst

    def PatientProc(self):
        mr = DatasetsProcessing.PreprocessingUtils.MRIProcessing(self.data) # type: ignore
        mr.Normalization8bit()
        mr.SampleWiseStandardization()
        mr.resize(self.ResSize,self.ResSize)
        self.data = mr.GetPatient()
        
    def MaskProc(self):
        mr = DatasetsProcessing.PreprocessingUtils.MaskProcessing(self.labels) # type: ignore
        mr.MaskIntegerization()
        mr.resize(self.ResSize,self.ResSize)
        self.labels = mr.GetMasks()    
        

    def GetProcessedPatients(self):
        return self.data

    def GetProcessedMasks(self):
        return self.labels