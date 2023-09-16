import numpy as np
import seg_metrics.seg_metrics as sg
import tensorflow as tf
import SimpleITK as sitk
import os
import SimpleITK as sitk

from ProstateLesionDetectionUtils.DetectionMetrics.Metric import DetorientMetric
from ProstateLesionDetectionUtils.DetectionMetrics.ModelDetoriention import DetorientLesions
from ProstateLesionDetectionUtils.Evaluation.Loss_functions import dice_coefficient

class Sensitivity_Analysis:
    def __init__(self, ytrue, ypred) -> None:
        self.ytrue = ytrue
        self.ypred = ypred
        self.ths = {}
    
    def extract_thresholds(self, num_thresholds = 20):
        """performs the sensitivity analysis for a patient and specific thresholds
        num_thresholds[int]: number of thresholds. Defaults to 20
        """
        thrs = np.linspace(0.0, 1, num = num_thresholds)
        tr = tf.cast(self.ytrue,tf.float32)
        for val in thrs:
            pred_done = np.where(self.ypred>val,1,0)
            prd= tf.cast(pred_done, tf.float32)
            self.ths.update({val: dice_coefficient(prd, tr).numpy()})
        return self.ths

class Model_Prediction:
    def __init__(self, parameters):
        """
        Constructor for the Evaluation Class.
        Parameters [dict]:  a dictionary containing all the evaluation parameters to execture the evaluation class
        """
        self.parameters = parameters
        self.predictions = {}
        self.mtr = {}

    def predict(self):
        
        for key in self.parameters["EVALUATION_DATA"].keys():
            t2 = sitk.GetArrayFromImage(self.parameters["EVALUATION_DATA"][key]["T2"])
            adc = sitk.GetArrayFromImage(self.parameters["EVALUATION_DATA"][key]["ADC"])
            dwi = sitk.GetArrayFromImage(self.parameters["EVALUATION_DATA"][key]["DWI"])
            stacked = np.transpose((t2, adc, dwi),(1,2,3,0))
            pred = sitk.GetImageFromArray(tf.squeeze(tf.squeeze(self.parameters["MODEL"].predict(tf.expand_dims(stacked, axis = 0)), axis= 0), axis=-1).numpy())
            pred.CopyInformation(self.parameters["EVALUATION_DATA"][key]["Lesions"])
            self.predictions.update({key: pred})
    
    def save_gt(self):
        try:
            os.mkdir(os.path.join(self.parameters["SAVE_FOLDER"],"Ground_Truth"))
        except:
            print("Directory exists")

        for key in self.parameters["EVALUATION_DATA"].keys():
            sitk.WriteImage(self.parameters["EVALUATION_DATA"][key]["Lesions"], os.path.join(os.path.join(self.parameters["SAVE_FOLDER"],"Ground_Truth"),"{}.nii.gz".format(key)))

    def save_predictions(self):
        try:
            os.mkdir(os.path.join(self.parameters["SAVE_FOLDER"],"predictions"))
        except:
            print("Directory exists")

        for key in self.parameters["EVALUATION_DATA"].keys():
            sitk.WriteImage(self.predictions[key], os.path.join(os.path.join(self.parameters["SAVE_FOLDER"],"predictions"),"{}.nii.gz".format(key)))


    def save_predictions_binary(self, threshold=0.5):
        try:
            os.mkdir(os.path.join(self.parameters["SAVE_FOLDER"],"Binary_Predictions"))
        except:
            print("Directory exists")

        for key in self.parameters["EVALUATION_DATA"].keys():
            vol = sitk.GetArrayFromImage(self.predictions[key])
            vol = np.where(vol>threshold, 1, 0).astype(np.uint8)
            vol_sitk = sitk.GetImageFromArray(vol)
            vol_sitk.CopyInformation(self.parameters["EVALUATION_DATA"][key]["Lesions"])
            sitk.WriteImage(vol_sitk, os.path.join(os.path.join(self.parameters["SAVE_FOLDER"],"Binary_Predictions"),"{}.nii.gz".format(key)))

class Evaluation:
    def __init__(self, parameters):
        """
        Constructor for the Evaluation Class.
        Parameters [dict]:  a dictionary containing all the evaluation parameters to execture the evaluation class
        """
        self.parameters = parameters
        self.predictions = {}
        self.mtr = {}

    def calculate_detorient_metric(self):

        patients_np = self.__convert_sitk_to_np(self.parameters["EVALUATION_DATA"])
        dt = DetorientLesions(patients_np, rates = [15,40,50])
        dt.CreateZones()
        dt.CreateZaxisZone()
        dt.KeepZonesNonZeroImageValues()
        zones = dt.GetZones()
        zaxis = dt.GetZaxisZones()

        preds_np= {}
        for item in self.predictions.keys():
            preds_np.update({item:sitk.GetArrayFromImage(self.predictions[item])})
        metric = DetorientMetric(np.array(list(zones.values())),np.array(list(zaxis.values())),np.array(list(preds_np.values())))
        metric.CalculateXYplane()
        metric.CalculateZplane()
        metric.ComputeMetric()
        mtr = metric.GetMetric()
        for det_val, pt_name in zip(list(mtr.values()),list(self.predictions.keys())):
            self.mtr.update({pt_name: {"Detorient_Metric":det_val}})

    def get_metrics(self, threshold=0.5):
        metrics = {}
        for key in self.parameters["EVALUATION_DATA"].keys():
            vol = sitk.GetArrayFromImage(self.predictions[key])
            vol = np.where(vol>threshold, 1, 0).astype(np.uint8)
            vol_sitk = sitk.GetImageFromArray(vol)
            vol_sitk.CopyInformation(self.parameters["EVALUATION_DATA"][key]["Lesions"])
            labels = [1]
            wm = sg.write_metrics(labels = labels,  # exclude background
                    gdth_img=self.parameters["EVALUATION_DATA"][key]["Lesions"],
                    pred_img=vol_sitk)
            tr = tf.cast(sitk.GetArrayFromImage(self.parameters["EVALUATION_DATA"][key]["Lesions"]),tf.float32)
            prd= tf.cast(sitk.GetArrayFromImage(self.predictions[key]), tf.float32)
            sna = Sensitivity_Analysis(sitk.GetArrayFromImage(self.parameters["EVALUATION_DATA"][key]["Lesions"]),
                                       sitk.GetArrayFromImage(self.predictions[key]))
            thrs = sna.extract_thresholds()

            wm[0].update({"Dice Volumetric":dice_coefficient(tr, prd).numpy(),
                          "Sensitivity Analysis": thrs})
            metrics.update({key:wm})
            if len(self.mtr)>0:
                metrics[key][0].update(self.mtr[key])
        return metrics

    
    def get_predictions(self):
        return self.predictions

    def save_gradcam(self):
        pass

    @staticmethod
    def __convert_sitk_to_np(sitk_dict):
        patients_np = {}
        for item in sitk_dict.keys():
            seqs = {}
            for key in ["T2","ADC","DWI","Lesions"]:
                seqs.update({key:sitk.GetArrayFromImage(sitk_dict[item][key])})
            patients_np.update({item:seqs})
        return patients_np