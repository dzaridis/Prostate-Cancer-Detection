import pickle

class PclLoader:
    def __init__(self, path158 = {}, pathpicai = {}, path_prnet = {}):
        """Constructor for pickle loader

        Args:
            path158 (str path file): pickle file to load
            pathpicai (path or list of paths): 
        """
        self.path158 = path158
        self.pathpicai = pathpicai
        self.path_prnet = path_prnet
    
    def P158Load(self):
        """Loader for the Prostate 158 dataset 

        Returns:
            [dict object]: dict object containing T2, ADC, DWI, Lesions, WG for each patient
        """
        with open(self.path158, 'rb') as handle:
            P158 = pickle.load(handle)
        return P158

    def ProstateNetLoad(self):
        """Loader for the ProstateNet dataset

        Returns:
            [dict object]: dict object containing T2, ADC, DWI, Lesions, WG for each patient
        """
        with open(self.path_prnet, 'rb') as handle:
            prnet = pickle.load(handle)
        return prnet
    
    def PicaiLoad(self):
        """Loader for the Picai dataset for all the folds [0-3]

        Returns:
            [dict object]: dict object containing T2, ADC, DWI, Lesions, WG for each patient
        """
        PICAI = {"T2":{},"ADC":{},"DWI":{},"Lesions":{},"WG":{}}
        for fold in self.pathpicai:
            with open(fold, 'rb') as handle:
                PicDict = pickle.load(handle)
            for key in PICAI.keys():
                PICAI[key].update(PicDict[key])
        return PICAI

    def PicaiLoadFold(self,fld=0):
        """Loader for the Picai dataset for a specific fold

        Returns:
            [dict object]: dict object containing T2, ADC, DWI, Lesions, WG for each patient
        """
        PICAI = {"T2":{},"ADC":{},"DWI":{},"Lesions":{},"WG":{}}
        with open(self.pathpicai[fld], 'rb') as handle:
            PicDict = pickle.load(handle)
        for key in PICAI.keys():
            PICAI[key].update(PicDict[key])
        return PICAI
