import SimpleITK as sitk, numpy as np, os

class LoadSitkImages:
    def __init__(self,ptdc, folder):
        """_summary_

        Args:
            ptdc (dictionary): Dictionary that contains patients with folded dictionary with sequences
            folder (path string): parent folder
        """
        self.ptdc = ptdc
        self.folder = folder
        self.stob = {}
        self.npob = {}
    
    def load_sitkobj(self):
        """returns the Sitk image Objects as a dictionary

        Returns:
            Dictionary: Dictionary that contains the Simple ITK Image Objects
        """
        for pat in self.ptdc.keys():
            t2  = sitk.ReadImage(os.path.join(self.folder, self.ptdc[pat]["T2"]))
            adc = sitk.ReadImage(os.path.join(self.folder, self.ptdc[pat]["ADC"]))
            dwi = sitk.ReadImage(os.path.join(self.folder, self.ptdc[pat]["DWI"]))
            les = sitk.ReadImage(os.path.join(self.folder, self.ptdc[pat]["Lesion"]))
            self.stob.update({pat:{"T2":t2,"ADC":adc,"DWI":dwi,"Lesions":les}})
        return self.stob

    def load_npobj(self):
        """returns the numpy arrays as a dictionary

        Returns:
            Dictionary: Dictionary that contains the Simple ITK Image Objects
        """
        for pat in self.ptdc.keys():
            t2 = sitk.ReadImage(os.path.join(self.folder, self.ptdc[pat]["T2"]))
            t2 = sitk.GetArrayFromImage(t2)
            adc = sitk.ReadImage(os.path.join(self.folder, self.ptdc[pat]["ADC"]))
            adc = sitk.GetArrayFromImage(adc)
            dwi = sitk.ReadImage(os.path.join(self.folder, self.ptdc[pat]["DWI"]))
            dwi = sitk.GetArrayFromImage(dwi)
            les = sitk.ReadImage(os.path.join(self.folder, self.ptdc[pat]["Lesion"]))
            les = sitk.GetArrayFromImage(les)
            self.npob.update({pat:{"T2":t2,"ADC":adc,"DWI":dwi,"Lesions":les}})
        return self.npob