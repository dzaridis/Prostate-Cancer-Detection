from ctypes import set_errno
import numpy as np, os
from medpy.io import load
import SimpleITK as sitk
import nibabel as nib
import random
from .Resampler import *

class Picai:
    def __init__(self, picai_path, labels_path):
        """Constructor for Picai class loader

        Args:
            picai_path (str path file): folder of the picai patients
            labels_path (str path file): folder of the annotation path
        """
        self.PicaiPath = picai_path
        self.LabelsPath = labels_path
        self.T2, self.labels = {}, {}
        self.ADC = {}
        self.DWI = {}

    def LoadT2(self):
        """
        Load T2 images and corresponding labels
        """
        t2w_pats = []
        labels = []

        for patient in (os.listdir(self.PicaiPath)):
            for sequence in (os.listdir(os.path.join(self.PicaiPath, patient))):
                if "t2w" in sequence:
                    t2w_pats.append(os.path.join(os.path.join(self.PicaiPath, patient, sequence)))

        for label in (os.listdir(self.LabelsPath)):
            labels.append(os.path.join(self.LabelsPath, label))

        # Crossmatching Annotation

        for l_p in labels:
            for d_p in t2w_pats:
                data_name = os.path.basename(os.path.normpath(d_p))
                labels_name = os.path.basename(os.path.normpath(l_p))

                if data_name[:13] == labels_name[:13]:
                    temp1 = sitk.ReadImage(d_p)
                    temp1 = sitk.DICOMOrient(temp1, 'LPS')
                    temp1 = sitk.GetArrayFromImage(temp1)

                    temp2 = sitk.ReadImage(l_p)
                    temp2 = sitk.DICOMOrient(temp2, 'LPS')
                    temp2 = sitk.GetArrayFromImage(temp2)
                    self.T2.update({data_name[:13]: temp1})
                    self.labels.update({labels_name[:13]: temp2})

    def LoadADC(self, register=True):
        """
        Load ADC Picai data (Patients in .mha and data in nii.gzz)
        Args:

        Returns:
            d : Dictionary of a patient's array
            l : Dictionary of a patient's label array
        """
        adc_pats = []
        labels = []
        t2w_pats = []

        for patient in (os.listdir(self.PicaiPath)):
            for sequence in (os.listdir(os.path.join(self.PicaiPath, patient))):
                if "adc" in sequence:
                    adc_pats.append(os.path.join(os.path.join(self.PicaiPath, patient, sequence)))
        if register:
            for patient in (os.listdir(self.PicaiPath)):
                for sequence in (os.listdir(os.path.join(self.PicaiPath, patient))):
                    if "t2w" in sequence:
                        t2w_pats.append(os.path.join(os.path.join(self.PicaiPath, patient, sequence)))

        for label in (os.listdir(self.LabelsPath)):
            labels.append(os.path.join(self.LabelsPath, label))

        # kanoume to cross match annotation data
        self.adc, csPCA = {}, {}
        for l_p in labels:
            for index, d_p in enumerate(adc_pats):
                if register:
                    t2_name = os.path.basename(os.path.normpath(t2w_pats[index]))
                adc_name = os.path.basename(os.path.normpath(d_p))
                labels_name = os.path.basename(os.path.normpath(l_p))
                if adc_name[:13] == labels_name[:13]:

                    temp1 = sitk.ReadImage(d_p)
                    temp1 = sitk.DICOMOrient(temp1, 'LPS')
                    temp11 = sitk.GetArrayFromImage(temp1)
                    if register:
                        tempt2 = sitk.ReadImage(t2w_pats[index])
                        tempt2 = sitk.DICOMOrient(tempt2, 'LPS')
                        tempt22 = sitk.GetArrayFromImage(tempt2)

                    temp2 = sitk.ReadImage(l_p)
                    temp2 = sitk.DICOMOrient(temp2, 'LPS')
                    temp2 = sitk.GetArrayFromImage(temp2)
                    if register:
                        ref, temp11 = ResampleImage_refT2(tempt2, temp1)

                    self.adc.update({adc_name[:13]: temp11})

    def LoadDWI(self, register=True):
        """
        Load Picai data (Patients in .mha and data in nii.gzz)
        Args:
            picai_path(str path object) : folder of the picai patients
            labels_path(str path object) : folder of the annotation path
        Returns:
            d : Dictionary of a patient's array
            l : Dictionary of a patient's label array
        """
        t2w_pats = []
        labels = []
        dwi_pats = []

        for patient in (os.listdir(self.PicaiPath)):
            for sequence in (os.listdir(os.path.join(self.PicaiPath, patient))):
                if "hbv" in sequence:
                    dwi_pats.append(os.path.join(os.path.join(self.PicaiPath, patient, sequence)))
        if register:
            for patient in (os.listdir(self.PicaiPath)):
                for sequence in (os.listdir(os.path.join(self.PicaiPath, patient))):
                    if "t2w" in sequence:
                        t2w_pats.append(os.path.join(os.path.join(self.PicaiPath, patient, sequence)))

        for label in (os.listdir(self.LabelsPath)):
            labels.append(os.path.join(self.LabelsPath, label))

        # perform to cross match annotation data
        self.dwi, csPCA = {}, {}
        for l_p in labels:
            for index, d_p in enumerate(dwi_pats):
                if register:
                    t2_name = os.path.basename(os.path.normpath(t2w_pats[index]))
                dwi_name = os.path.basename(os.path.normpath(d_p))
                labels_name = os.path.basename(os.path.normpath(l_p))

                if dwi_name[:13] == labels_name[:13]:

                    temp1 = sitk.ReadImage(d_p)
                    temp1 = sitk.DICOMOrient(temp1, 'LPS')
                    temp11 = sitk.GetArrayFromImage(temp1)
                    if register:
                        tempt2 = sitk.ReadImage(t2w_pats[index])
                        tempt2 = sitk.DICOMOrient(tempt2, 'LPS')
                        tempt22 = sitk.GetArrayFromImage(tempt2)

                    temp2 = sitk.ReadImage(l_p)
                    temp2 = sitk.DICOMOrient(temp2, 'LPS')
                    temp2 = sitk.GetArrayFromImage(temp2)
                    if register:
                        ref, temp11 = ResampleImage_refT2(tempt2, temp1)

                    self.dwi.update({dwi_name[:13]: temp11})

    def GetT2(self):
        """Get the T2 dictionaries

        Returns:
            T2 (dictionary): keys are the names values are the T2 arrays
        """
        return self.T2

    def GetADC(self):
        """Get the adc dictionaries

        Returns:
            ADC (dictionary): keys are the names values are the ADC arrays
        """
        return self.adc

    def GetDWI(self):
        """Get the DWI dictionaries

        Returns:
            DWI (dictionary): keys are the names values are the DWI arrays
        """
        return self.dwi

    def GetLabels(self):
        """Get the labels dictionaries

        Returns:
            labels (dictionary): keys are the names values are the Labels arrays
        """
        return self.labels


class KeepVolumeMetaPicai:
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.ITKimages = {}
        self.meta = {}

    def load_sitk_images(self, seq="t2w"):
        """
        Loads sitk image objects from filepath
        Args:
            seq [str]: sequence to choose, t2w, adc, hbv
        """
        for patient in os.listdir(self.filepaths):
            for sequence in (os.listdir(os.path.join(self.filepaths, patient))):
                if seq in sequence:
                    pth = os.path.join(os.path.join(self.filepaths, patient), sequence)
                    print(pth)
                    ITK_image = sitk.ReadImage(pth)
                    ITK_image = sitk.DICOMOrient(ITK_image, 'LPS')

                    self.ITKimages.update({sequence[:13]: ITK_image})

    def load_metadata(self):
        """
        Loads origin, spacing and direction into a dictionary from sitk image object
        """
        for patname in self.ITKimages.keys():
            self.meta.update({patname: {"Spacing": self.ITKimages[patname].GetSpacing(),
                                        "Origin": self.ITKimages[patname].GetOrigin(),
                                        "Direction": self.ITKimages[patname].GetDirection(),
                                        "PixelID": self.ITKimages[patname].GetPixelIDValue(),
                                        "Size":self.ITKimages[patname].GetSize()}})

    def get_sitk_images(self):
        """
        Returns the sitk image object in a dictionary
        Returns:
            [dict]: keys are the names values are the sitk image objects for the selected sequence
        """
        return self.ITKimages

    def get_meta(self):
        """
        Returns the sitk image metadata namely, spacing, origin, direction for each patient
        Returns:
            [dict]: keys are the names values are the sitk image objects for the selected sequence
        """
        return self.meta
    
class Prostate158:
    def __init__(self, pat_path):
        self.pat_path = pat_path
        self.T2 = {}
        self.ADC = {}
        self.DWI = {}
        self.WG = {}
        self.PZ = {}
        self.TZ = {}
        self.LesionADC = {}
        self.LesionT2 = {}

    def LoadT2(self):
        for patient in os.listdir(self.pat_path):
            series = os.path.join(self.pat_path, patient)
            for ser in os.listdir(series):
                if ser == "t2.nii.gz":
                    img = nib.load(os.path.join(series, ser)).get_fdata()
                    img = np.transpose(img, (2, 1, 0))
                    self.T2.update({patient: img})

    def LoadADC(self):
        for patient in os.listdir(self.pat_path):
            series = os.path.join(self.pat_path, patient)
            for ser in os.listdir(series):
                if ser == "adc.nii.gz":
                    img = nib.load(os.path.join(series, ser)).get_fdata()
                    img = np.transpose(img, (2, 1, 0))
                    self.ADC.update({patient: img})

    def LoadDWI(self):
        for patient in os.listdir(self.pat_path):
            series = os.path.join(self.pat_path, patient)
            for ser in os.listdir(series):
                if ser == "dwi.nii.gz":
                    img = nib.load(os.path.join(series, ser)).get_fdata()
                    img = np.transpose(img, (2, 1, 0))
                    self.DWI.update({patient: img})

    def LoadAnatomies(self):
        for patient in os.listdir(self.pat_path):
            series = os.path.join(self.pat_path, patient)
            for ser in os.listdir(series):
                if "anatomy" in ser:
                    img = nib.load(os.path.join(series, ser)).get_fdata()
                    img = np.transpose(img, (2, 1, 0))
                    per = np.where(img == 2., 1, 0).astype(int)
                    tra = np.where(img == 1., 1, 0).astype(int)
                    wg = np.where(img == 0., 0, 1).astype(int)

                    self.WG.update({patient: wg})
                    self.PZ.update({patient: wg})
                    self.TZ.update({patient: wg})

    def LoadLesionADC(self):
        for patient in os.listdir(self.pat_path):
            series = os.path.join(self.pat_path, patient)
            cnt = 0
            for ser in os.listdir(series):
                if ser == "adc_tumor_reader1.nii.gz":
                    img = nib.load(os.path.join(series, ser)).get_fdata()
                    img = np.transpose(img, (2, 1, 0))
                    img = np.where(img > .9, 1, 0).astype(int)
                    self.LesionADC.update({patient: img})
                    cnt += 1

    def LoadLesionT2(self):
        for patient in os.listdir(self.pat_path):
            series = os.path.join(self.pat_path, patient)
            cnt = 0
            for ser in os.listdir(series):
                if ser == "t2_tumor_reader1.nii.gz":
                    img = nib.load(os.path.join(series, ser)).get_fdata()
                    img = np.transpose(img, (2, 1, 0))
                    img = np.where(img > .9, 1, 0).astype(int)
                    self.LesionT2.update({patient: img})
                    cnt += 1

    def GetT2(self):
        return self.T2

    def GetADC(self):
        return self.ADC

    def GetDWI(self):
        return self.DWI

    def GetLesionADC(self):
        return self.LesionADC

    def GetLesionT2(self):
        return self.LesionT2

    def GetWG(self):
        return self.WG

    def GetPZ(self):
        return self.PZ

    def GetTZ(self):
        return self.TZ


class KeepVolumeMetaP158(KeepVolumeMetaPicai):
    def __init__(self, filepaths):
        super().__init__(filepaths)
        self.filepaths = filepaths
        self.ITKimages = {}
        self.meta = {}

    def load_sitk_images(self):
        """
        Loads sitk image objects from filepath
        """
        for patient in os.listdir(self.filepaths):
            series = os.path.join(self.filepaths, patient)
            for ser in os.listdir(series):
                if ser == "t2.nii.gz":
                    ITK_image = sitk.ReadImage(os.path.join(series, ser))
                    ITK_image = sitk.DICOMOrient(ITK_image, 'LPS')

            self.ITKimages.update({patient: ITK_image})