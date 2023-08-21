import numpy as np
class DetorientMetric:
    def __init__(self,SpatZones_val,ZaxisZones_val,PredsNormusenet):
        """Detorient metric

        Args:
            SpatZones_val (4D np.array): The xy zones to compute the spatial metric
            ZaxisZones_val (4D np.array): The z zone to compute the spatial metric
            PredsNormusenet (4D np.array): Normalized in 0-1 soft prediction by the model
        """
        self.SpatZones_val = SpatZones_val
        self.ZaxisZones_val = ZaxisZones_val
        self.PredsNormusenet = PredsNormusenet
        self.detxy = {}
        self.detz = {}
        self.metric = {}

    def CalculateXYplane(self):
        """Computes the z plane detorient metric"""
        lab_vals = [3,2,1,0]
        for pat in range(self.SpatZones_val.shape[0]):
            Detorient = []
            for ind,slice in enumerate(self.SpatZones_val[pat,:,:,:]):
                W1 = {}
                #print (np.unique(slice).shape[0])
                if np.unique(slice).shape[0] > 1:
                    for zn in lab_vals:
                        a = np.where(slice == zn, self.PredsNormusenet[pat,ind,:,:], 0)
                        ls = [val for val in a.flatten() if val>0.001]
                        if zn == 3:
                            try:
                                W1.update({"ROI":1*sum(ls)/len(ls)})
                            except:
                                W1.update({"ROI":0})
                        elif zn == 2:
                            try:
                                W1.update({"Zone1":-0.2*sum(ls)/len(ls)})
                            except:
                                W1.update({"Zone1":0})
                        elif zn == 1:
                            try:
                                W1.update({"Zone2":-0.3*sum(ls)/len(ls)})
                            except:
                                W1.update({"Zone2":0})
                        elif zn == 0:
                            try:
                                W1.update({"Zone3":-0.5*sum(ls)/len(ls)})
                            except:
                                W1.update({"Zone3":0})
                    Detorient.append(W1["ROI"]+W1["Zone1"]+W1["Zone2"]+W1["Zone3"])
            self.detxy.update({pat:sum(Detorient)/len(Detorient)})
    
    def CalculateZplane(self):
        """Computes the z plane detorient metric
        """
        zones = [2,1,0]
        weights = [1,-0.25,-0.75]
        for pat in range(self.ZaxisZones_val.shape[0]):
            Wz = []
            for ind,slice in enumerate(self.ZaxisZones_val[pat,:,:,:]):
                for zone,weight in zip(zones,weights):
                    if np.unique(slice)[0] == zone:
                        ls = [val for val in self.PredsNormusenet[pat,ind,:,:].flatten() if val>0.001]
                        Wz.append(weight*sum(ls)/len(ls))
            #zplanedet = sum(Wz)/len(Wz)
            self.detz.update({pat:sum(Wz)/len(Wz)})
    
    def ComputeMetric(self):
        """computes the average between xy and z planes 
        """
        for key in self.detz.keys():
            self.metric.update({key:(self.detz[key]+self.detxy[key])/2})
    
    def GetXYandZplanes(self):
        return self.detxy, self.detz
    
    def GetMetric(self):
        return self.metric
