import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import gridspec
import SimpleITK as sitk


# class VisualizationSeries:
#     def __init__(self, seq_dict):
#         self.seq_dict = seq_dict

#     def show_patient_4D(self, PAT, IsSitkIm=True, alpha=10):
#         """
#         Visualization of a specific patient
#         Args:
#             seq_dict [dict]:  contains sequences as keys and all patients for each key. Expects keys to be "T2","ADC","DWI",
#             "Lesions","WG"
#             PAT [str]: patient ID
#             IsSitkIm [boolean]: whether is a sitk image oject or numpy array
#             alpha [int]: to set the power of the mask to overlap

#         """
#         print("Patient{}".format(PAT))
#         if IsSitkIm:
#             plt.figure()
#             fig = px.imshow(sitk.GetArrayFromImage(self.seq_dict['T2'][PAT][:, :, :, 0]) + alpha * sitk.GetArrayFromImage(
#                 self.seq_dict['Lesions'][PAT][:, :, :])
#                             , animation_frame=0, binary_string=True)
#             fig.show()
#             plt.figure()
#             fig = px.imshow(sitk.GetArrayFromImage(self.seq_dict['ADC'][PAT][:, :, :]), animation_frame=0,
#                             binary_string=True)
#             fig.show()
#             plt.figure()
#             fig = px.imshow(sitk.GetArrayFromImage(self.seq_dict['DWI'][PAT][:, :, :]), animation_frame=0,
#                             binary_string=True)
#             fig.show()

#         else:
#             plt.figure()
#             fig = px.imshow(self.seq_dict['T2'][PAT][:, :, :, 0]+alpha*self.seq_dict['Lesions'][PAT][:, :, :, 0]
#                             , animation_frame=0, binary_string=True)
#             fig.show()
#             plt.figure()
#             fig = px.imshow(self.seq_dict['ADC'][PAT][:, :, :, 0], animation_frame=0, binary_string=True)
#             fig.show()
#             plt.figure()
#             fig = px.imshow(self.seq_dict['DWI'][PAT][:, :, :, 0], animation_frame=0, binary_string=True)
#             fig.show()


#     def show_patient_3D(self, PAT, IsSitkIm=True, alpha=10):
#         """
#         Visualization of a specific patient
#         Args:
#             seq_dict [dict]:  contains sequences as keys and all patients for each key. Expects keys to be "T2","ADC","DWI",
#             "Lesions","WG"
#             PAT [str]: patient ID
#             IsSitkIm [boolean]: whether is a sitk image oject or numpy array
#             alpha [int]: to set the power of the mask to overlap

#         """
#         print("Patient{}".format(PAT))
#         if IsSitkIm:
#             plt.figure()
#             fig = px.imshow(sitk.GetArrayFromImage(self.seq_dict['T2'][PAT][:, :, :]) + alpha * sitk.GetArrayFromImage(
#                 self.seq_dict['Lesions'][PAT][:, :, :]) , animation_frame=0, binary_string=True)
#             fig.show()
#             plt.figure()
#             fig = px.imshow(sitk.GetArrayFromImage(self.seq_dict['ADC'][PAT][:, :, :]), animation_frame=0,
#                             binary_string=True)
#             fig.show()
#             plt.figure()
#             fig = px.imshow(sitk.GetArrayFromImage(self.seq_dict['DWI'][PAT][:, :, :]), animation_frame=0,
#                             binary_string=True)
#             fig.show()

#         else:
#             plt.figure()
#             fig = px.imshow(self.seq_dict['T2'][PAT][:, :, :]+alpha*self.seq_dict['Lesions'][PAT][:, :, :]
#                             , animation_frame=0, binary_string=True)
#             fig.show()
#             plt.figure()
#             fig = px.imshow(self.seq_dict['ADC'][PAT][:, :, :], animation_frame=0, binary_string=True)
#             fig.show()
#             plt.figure()
#             fig = px.imshow(self.seq_dict['DWI'][PAT][:, :, :], animation_frame=0, binary_string=True)
#             fig.show()

# def show_patient_and_zones(self,zone_dict, PAT, IsSitkIm=True, alpha=10):
#     """
#     Visualization of a specific patient
#     Args:
#         seq_dict [dict]:  contains sequences as keys and all patients for each key. Expects keys to be "T2","ADC","DWI",
#         "Lesions","WG"
#         PAT [str]: patient ID
#         IsSitkIm [boolean]: whether is a sitk image oject or numpy array
#         alpha [int]: to set the power of the mask to overlap

#     """
#     print("Patient{}".format(PAT))
#     if IsSitkIm:
#         plt.figure()
#         fig = px.imshow(sitk.GetArrayFromImage(self.seq_dict['T2'][PAT][:, :, :,0]) + alpha * sitk.GetArrayFromImage(
#             self.seq_dict['Lesions'][PAT][:, :, :])
#                         , animation_frame=0, binary_string=True)
#         fig.show()
#         plt.figure()
#         fig = px.imshow(sitk.GetArrayFromImage(self.seq_dict['ADC'][PAT][:, :, :,0]), animation_frame=0,
#                         binary_string=True)
#         fig.show()
#         plt.figure()
#         fig = px.imshow(sitk.GetArrayFromImage(self.seq_dict['DWI'][PAT][:, :, :,0]), animation_frame=0,
#                         binary_string=True)
#         fig.show()

#     else:
#         plt.figure()
#         fig = px.imshow(self.seq_dict['T2'][PAT][:, :, :, 0] + 0.5*zone_dict["XY"][PAT][:, :, :, 0]
#                         +.3*zone_dict["Z"][PAT][:, :, :, 0], animation_frame=0, binary_string=True)
#         fig.show()
#         plt.figure()
#         fig = px.imshow(self.seq_dict['ADC'][PAT][:, :, :, 0], animation_frame=0, binary_string=True)
#         fig.show()
#         plt.figure()
#         fig = px.imshow(self.seq_dict['DWI'][PAT][:, :, :, 0], animation_frame=0, binary_string=True)
#         fig.show()



class Visualizer():
    def __init__(self,patdict):
        self.patdict = patdict

    def np_vis(self, sequence, patient, alpha = 0.5):
        fig = plt.figure(figsize=(24,24))
        gs = gridspec.GridSpec(nrows=3, ncols=8, wspace=0.5, hspace=0.5)

        for i in range(24):
            ax = plt.subplot(4, 6,i + 1)
            ax.imshow(self.patdict[patient][sequence][i, :,:]+self.patdict[patient]["Lesions"][i, :,:]*alpha, cmap="gray")
            ax.set_title("Slice "+str(i), fontsize=14)
            ax.axis('off')  # Hide axes for a cleaner look

        plt.suptitle('MRI Slices {sequence} and {patient}', fontsize=20, y=0.92)  # Add a centered title to the figure
        plt.show()

    def sitk_vis(self, sequence, patient, alpha=0.5):
        fig = plt.figure(figsize=(24,24))
        gs = gridspec.GridSpec(nrows=3, ncols=8, wspace=0.5, hspace=0.5)

        for i in range(24):
            ax = plt.subplot(4, 6,i + 1)
            ax.imshow(sitk.GetArrayFromImage(self.patdict[patient][sequence][i, :,:])+sitk.GetArrayFromImage(self.patdict[patient]["Lesions"][i, :,:])*alpha, cmap="gray")
            ax.set_title("Slice "+str(i), fontsize=14)
            ax.axis('off')  # Hide axes for a cleaner look

        plt.suptitle('MRI Slices {sequence} and {patient}', fontsize=20, y=0.92)  # Add a centered title to the figure
        plt.show()