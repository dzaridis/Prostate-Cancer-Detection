import numpy as np
import ProstateLesionDetectionUtils


class DetorientLesions:
    def __init__(self,patients, rates = [15,40,50]) -> None:
        """Constructor for the Detorient Deep learning models metric
        Args:
            PatDict (dict): Dictionary with keys as patient's names and values 3D image arrays
            LesionsDict (dict): dictionary containing patient names as keys and corresponding binary masks as values
            rates (list, optional): dilation rates to create the zones. Defaults to [15,40,50].
        """
        self.patients = patients
        self.rates = rates
        self.LesionsDict = {}
        for pat in patients.keys():
            self.LesionsDict.update({pat : patients[pat]["Lesions"]})
        self.zones = {}
        self.Zaxis = {}
    @staticmethod
    def CreateDils(LesionsDict, rate):
        """Static class method to dilate the mask and return the dilated dictionary 

        Args:
            LesionsDict (dictionary): dictionary containing patient names as keys and corresponding binary masks as values
            rate (int): [dilation rate to extend the mask

        Returns:
            [dict]: dictionary with the dilated binary mask
        """
        maskproc = ProstateLesionDetectionUtils.DatasetsProcessing.PreprocessingUtils3D.MaskProcessing(LesionsDict)
        maskproc.MaskIntegerization()
        maskproc.dilation_alg(rate)
        dict = maskproc.GetMasks()
        return dict
    
    def CreateZones(self) -> None:
        """
        Creates different zones in respect to the distance from the Gt mask
        """
        lsdcs = []
        for rate in self.rates:
            lsdcs.append(self.CreateDils(self.LesionsDict, rate))
        for key in self.LesionsDict.keys():
            base = np.where(self.LesionsDict[key]>0.5,1,0)
            a = np.where(lsdcs[0][key]>.5,1,0)
            b = np.where(lsdcs[1][key]>.5,1,0)
            c = np.where(lsdcs[2][key]>.5,0,0)
            self.zones.update({key:base+a+b+c})

    def CreateZaxisZone(self):
        """Creates zones for z axis
        """
        for key in self.LesionsDict.keys():
            arr = np.zeros((self.LesionsDict[key].shape),dtype=int)
            for ind in range(self.LesionsDict[key].shape[0]):
                if np.unique(self.LesionsDict[key][ind]).shape[0]==2:
                    arr[ind].fill(2)
            try:
                arr[np.unique(np.where(arr == 2)[0]).min()-3:np.unique(np.where(arr == 2)[0]).min()].fill(1)
            except:
                arr[0:np.unique(np.where(arr == 2)[0]).min()].fill(1)
            try:
                arr[np.unique(np.where(arr == 2)[0]).max()+1:np.unique(np.where(arr == 2)[0]).max()+4].fill(1)
            except:
                arr[np.unique(np.where(arr == 2)[0]).max()+1:].fill(1)
            self.Zaxis.update({key:arr})

    def KeepZonesNonZeroImageValues(self) -> None:
        """Creates the zones only in the organ of interest
        """
        for key in self.patients.keys():
            self.zones.update({key:np.where(self.patients[key]["T2"]>0.001,self.zones[key],0)})
    
    def GetZones(self):
        return(self.zones)

    def GetZaxisZones(self):
        return (self.Zaxis)
