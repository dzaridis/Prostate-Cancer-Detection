import tensorflow as tf
import numpy as np
from skimage.morphology import dilation

import ProstateLesionDetectionUtils


class MRIProcessing:
    def __init__(self, PatDict) -> None:
        self.PatDict = PatDict

    def Normalization8bit(self):
        """
        Converts the images to have pixels' intensities in the range of 0-255
        """
        PatInst = {}
        for key in self.PatDict.keys():
            arr = [((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8) if (
                                                                                                   img.max() - img.min()) != 0 else np.nan_to_num(
                img, nan=0).astype(np.uint8)
                   for img in self.PatDict[key]]
            PatInst.update({key: np.asarray(arr)})
        self.PatDict = PatInst

    def SampleWiseStandardization(self):
        """
        Normalize images into 0-1 
        """
        PatInst = {}
        for key in self.PatDict.keys():
            arr = [(img - img.min()) / (img.max() - img.min()) if (img.max() - img.min()) != 0 else np.nan_to_num(img,
                                                                                                                  nan=0).astype(
                np.uint8)
                   for img in self.PatDict[key]]
            PatInst.update({key: np.asarray(arr)})
        self.PatDict = PatInst

    def SampleWiseStandardization_Zscore(self):
        """
        Normalize images into 0-1
        """
        PatInst = {}
        for key in self.PatDict.keys():
            arr = [(img - np.mean(img))/np.std(img) if np.std(img) != 0 else np.nan_to_num(img,nan=0).astype(np.uint8)
                   for img in self.PatDict[key]]
            PatInst.update({key: np.asarray(arr)})
        self.PatDict = PatInst

    def PatientWiseStandardization(self):
        """
        Normalize Patients into 0-1  intensity scaling
        """
        PatInst = {}
        for key in self.PatDict.keys():
            arr = [(img - self.PatDict[key].min()) / (self.PatDict[key].max() - self.PatDict[key].min()) for img in
                   self.PatDict[key]]
            PatInst.update({key: np.asarray(arr)})
        self.PatDict = PatInst

    def resize(self, x, y) -> None:
        """ Resize the patients to certain dimensions on width and height with bilinear interpolation
        Args:
            x (int):Width to resize
            y (int):Height to resize
        """
        PatInst = {}
        for key in self.PatDict.keys():
            arr = [tf.image.resize(tf.expand_dims(img, axis=2), (256, 256), tf.image.ResizeMethod.BILINEAR) for img in
                   self.PatDict[key]]
            PatInst.update({key: np.array(arr)})
        self.PatDict = PatInst

    def GetPatient(self):
        return self.PatDict


class MaskProcessing:
    def __init__(self, MaskDict):
        self.MaskDict = MaskDict

    def resize(self, x, y) -> None:
        """ Resize the patients to certain dimensions on width and height with bilinear interpolation
        Args:
            x (int):Width to resize
            y (int):Height to resize
        """
        PatInst = {}
        for key in self.MaskDict.keys():
            arr = [tf.image.resize(tf.expand_dims(img, axis=2), (256, 256), tf.image.ResizeMethod.NEAREST_NEIGHBOR) for
                   img in self.MaskDict[key]]
            PatInst.update({key: np.array((arr))})
        self.MaskDict = PatInst

    def ObscureTinyMaskParts(self):
        """Obscures(blackens) very tiny masks tha might be issues on the manual segmentations
        """
        PatsMiss = []
        for key in self.MaskDict.keys():
            cnt = 0
            for ind, img in enumerate(self.MaskDict[key]):
                if np.unique(img, return_counts=True)[1].shape[0] == 2:
                    if np.unique(img, return_counts=True)[1][1] < 5:
                        if cnt == 0:
                            PatsMiss.append(key)
                        self.MaskDict[key][ind] = np.where(img == 1, 0, 0)
                    cnt += 1
        return PatsMiss

    def MaskIntegerization(self) -> None:
        """
        Processing the mask to have binary values in integer format 0-1
        """
        PatInst = {}
        for key in self.MaskDict.keys():
            processed_slice = []
            for img in self.MaskDict[key]:
                if np.max(img) != np.min(img):
                    img = np.where(img == np.max(img), 1, 0)
                    img = img.astype(int)
                else:
                    img = img.astype(int)
                processed_slice.append(img)
            PatInst.update({key: np.array(processed_slice)})
        self.MaskDict = PatInst

    def dilation_alg(self, iters) -> None:
        """ applies  dilation around the mask on annotations
        Args:
            iters(int): how many times the dilation will be applied
        """
        pats = {}
        for pat in self.MaskDict.keys():
            dilated_slices = []
            for img in self.MaskDict[pat]:
                im = img.astype(int)
                mask = dilation(im).astype(int)
                temp = mask
                for k in range(iters):
                    mask2 = dilation(temp)
                    temp = mask2
                dilated_slices.append(temp)
            dilated_slices = np.asarray(dilated_slices)
            pats.update({pat: dilated_slices})
        self.MaskDict = pats

    def crop_white_sides(self, proportion_to_black=0.1):
        """
        Fixes the sides of the mask, if white stripes appear at the sides of the image
        Args:
            proportion_to_black [float]: takes values from 0 to 1 and sets
            the proportion of image that needs to be fixed. I.E. i an image has shape of 256 and proportion_to_black=0.1
            then the first 25 pixels in xy are going to be fixed
        """
        pats = {}
        for pat in self.MaskDict.keys():
            fixed = []
            for img in self.MaskDict[pat]:
                im = img.astype(int)
                im[0:int(im.shape[0]*proportion_to_black), :] = 0
                im[:, 0:int(im.shape[1]*proportion_to_black)] = 0
                im[int(im.shape[0] * (1-proportion_to_black)):im.shape[0], :] = 0
                im[:, int(im.shape[1] * (1 - proportion_to_black)):im.shape[1]] = 0
                fixed.append(im)
            fixed = np.asarray(fixed)
            pats.update({pat: fixed})
        self.MaskDict = pats

    def Mask_Minus_DilatedMask(self, iters) -> None:
        """ applies  dilation around the mask on annotations and substract original mask from the dilated one
        Args:
            iters(int): how many times the dilation will be applied
        """
        pats = {}
        for pat in self.MaskDict.keys():
            dilated_slices = []
            for img in self.MaskDict[pat]:
                im = img.astype(int)
                mask = dilation(im).astype(int)
                temp = mask
                for k in range(iters):
                    mask2 = dilation(temp)
                    temp = mask2
                dilated_slices.append(abs(im - temp))
            dilated_slices = np.asarray(dilated_slices)
            pats.update({pat: dilated_slices})
        self.MaskDict = pats

    def GetMasks(self):
        return self.MaskDict


class PatientPad:
    def __init__(self, patient, WG):
        self.patient = patient
        self.WG = WG
        self.MinMaxValues = {}
        self.padded = {}
        self.padz = {}
        self.cropped = {}

    def ProstatePad(self):
        """
        Segments the WG from MRI volumes
        (blackens the area outside the prostate and keeps the prostate)
        """
        for pat in self.patient.keys():
            g = np.where(self.WG[pat] == 1, self.patient[pat], 0)
            self.patient[pat] = g

    def FindMask(self):
        """
        Tracks the minimum and maximum pixel location of the mask and store it into a dictionary
        for each patient
        """
        for key in self.WG.keys():
            lsvals = {}
            for ind, im in enumerate(self.WG[key]):
                if np.where(self.WG[key][ind] == 1)[0].shape[0] != 0:
                    xmax = np.max(np.where(self.WG[key][ind] == 1)[0])
                    xmin = np.min(np.where(self.WG[key][ind] == 1)[0])
                    ymax = np.max(np.where(self.WG[key][ind] == 1)[1])
                    ymin = np.min(np.where(self.WG[key][ind] == 1)[1])
                else:
                    xmax = im.shape[0]
                    xmin = 0
                    ymax = im.shape[1]
                    ymin = 0
                lsvals.update({ind: {"xmax": xmax, "xmin": xmin, "ymax": ymax, "ymin": ymin}})
            self.MinMaxValues.update({key: lsvals})

    def GetMinMaxValues(self):
        return self.MinMaxValues

    def CropWG(self, dims):
        """
        Crops prostate's WG and pads with zeros till the desired dims.
        E.G. Shall the dims be 256 and the cropped WG is 125x130, the image is padded 131 pixels in x dim and 126
        in y centering the prostate
        Args:
            dims[int]: dimensions to pad the image

        """
        for key in self.patient.keys():
            ims = np.zeros((self.patient[key].shape[0], dims, dims, 1))
            for ind, im in enumerate(self.patient[key]):
                try:
                    im1 = im[self.MinMaxValues[key][ind]["xmin"]:self.MinMaxValues[key][ind]["xmax"],
                         self.MinMaxValues[key][ind]["ymin"]:self.MinMaxValues[key][ind]["ymax"], 0]
                    diffx = self.MinMaxValues[key][ind]["xmax"] - self.MinMaxValues[key][ind]["xmin"]
                    diffy = self.MinMaxValues[key][ind]["ymax"] - self.MinMaxValues[key][ind]["ymin"]
                    padx = dims - diffx
                    pady = dims - diffy
                    if padx // 2 == padx / 2 and pady // 2 == pady / 2:
                        padded_image = np.pad(im1, ((padx // 2, padx // 2), (pady // 2, pady // 2)),
                                              constant_values=(0, 0))
                    elif padx // 2 != padx / 2 and pady // 2 == pady / 2:
                        padded_image = np.pad(im1, ((padx // 2, (padx // 2) + 1), (pady // 2, pady // 2)),
                                              constant_values=(0, 0))
                    elif padx // 2 == padx / 2 and pady // 2 != pady / 2:
                        padded_image = np.pad(im1, ((padx // 2, padx // 2), (pady // 2, (pady // 2) + 1)),
                                              constant_values=(0, 0))
                    else:
                        padded_image = np.pad(im1, ((padx // 2, (padx // 2) + 1), (pady // 2, (pady // 2) + 1)),
                                              constant_values=(0, 0))
                    padded_image = np.expand_dims(padded_image, axis=-1)
                except ValueError: # issues with the masks
                    padded_image = np.zeros(shape=(dims, dims, 1))
                ims[ind] = padded_image
            self.cropped.update({key: ims})


    def PadZaxis(self, dimz=24):
        """
        Pads or Crops each patient in the Z-axis (depth axis) if the patient has more slices than the desired crops the
        difference from the front and the end of the sequence (It is ensured that no information will be lost just null
        slices). On the contrary, if  the patient has less slices than the desired ones it pads zero-valued slices at
        the end and front
        Args:
            dimz [int]: desired dimensions to Z axis


        """

        for key in self.cropped.keys():
            sizez = self.cropped[key].shape[0]
            dif = sizez - dimz

            if dif > 0 and dif/2 == dif//2:
                self.padz.update({key: self.cropped[key][dif//2:sizez-dif//2, :, :, :]})
            elif dif > 0 and dif/2 != dif//2:
                self.padz.update({key: self.cropped[key][dif//2+1:sizez-dif//2, :, :, :]})
            elif dif < 0 and dif/2 == dif//2:
                pad_length = np.abs(dif)
                self.padz.update({key: np.pad(self.cropped[key],
                                              ((pad_length//2, pad_length//2), (0, 0), (0, 0), (0, 0)),
                                              mode='constant')})
            elif dif < 0 and dif/2 != dif//2:
                pad_length = np.abs(dif)
                self.padz.update({key: np.pad(self.cropped[key],
                                              ((pad_length//2+1, pad_length//2), (0, 0), (0, 0), (0, 0)),
                                              mode='constant')})
            else:
                self.padz.update({key: self.cropped[key]})

    def GetPaddedZaxis(self):
        return self.padz

    def GetCropped(self):
        return self.cropped

    def GetPadded(self):
        return self.padded


class PrepareDatasetForTraining:
    def __init__(self, patient_sequence, key="T2"):
        """

        Args:
            patient_sequence [dict]: contains all the sequences and patients for each one. It is expected
            to include "T2", "ADC", "DWI", "Lesions", "WG"
            key [str]: Specific sequence to select and prepare
        """
        self.patient_sequence = patient_sequence
        self.key = key
        self.Lesions = {}
        self.sequence = {}

    def keep_lesion_patients(self):
        """
        Keeps only patients with lesion masks and discard these without mask
        """
        lesion_pats = {}
        lesions = {}
        for key_pat in self.patient_sequence['Lesions'].keys():
            if self.patient_sequence['Lesions'][key_pat].any() > 0:
                lesion_pats.update({key_pat: self.patient_sequence[self.key][key_pat]})
                lesions.update({key_pat: self.patient_sequence["Lesions"][key_pat]})
        self.patient_sequence[self.key] = lesion_pats
        self.patient_sequence["Lesions"] = lesions

    def standardize(self):
        """
        Standardizes the images to have their pixels in the 0-1 range taking into consideration the Whole Gland.
        """
        proc = ProstateLesionDetectionUtils.DatasetsProcessing.PreprocessingUtils3D.MRIProcessing(
            self.patient_sequence[self.key])
        proc.SampleWiseStandardization_Zscore()
        self.patient_sequence[self.key] = proc.GetPatient()
        for pat in self.patient_sequence[self.key].keys():
            for ind, image in enumerate(self.patient_sequence[self.key][pat]):
                self.patient_sequence[self.key][pat][ind] = np.where(image < 0.1, 0, image)

    def fix_intensities(self):
        """
        Standardizes the images to have their pixels in the 0-1 range taking into consideration the Whole Gland.
        Pixel inside WG will be scaled at 0-1 range while pixels outside WG will be set to 0 value
        """
        for key in self.patient_sequence[self.key].keys():
            arr = [np.where(mask > 0.1, (img-np.min(img))/(np.max(img)-np.min(img)), 0) if (img.max() - img.min()) != 0
                   else np.nan_to_num(img,nan=0).astype(np.uint8)
                   for img, mask in zip(self.patient_sequence[self.key][key], self.patient_sequence["WG"][key])]
            self.patient_sequence[self.key].update({key: np.asarray(arr)})
        self.patient_sequence.update({self.key: self.patient_sequence[self.key]})

    def convert_mask_to_binary(self):
        """
        Converts lesion masks to be binary and type np.uint8
        """
        msk = ProstateLesionDetectionUtils.DatasetsProcessing.PreprocessingUtils3D.MaskProcessing(
            self.patient_sequence['Lesions'])
        msk.MaskIntegerization()
        self.patient_sequence["Lesions"] = msk.GetMasks()

    def get_processed_patients(self):
        """

        Returns: the training ready dataset

        """
        return self.patient_sequence
