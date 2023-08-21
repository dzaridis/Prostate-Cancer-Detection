import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import gridspec
import SimpleITK as sitk

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
