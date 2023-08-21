
import numpy as np, os
from medpy.io import load
import SimpleITK as sitk
import nibabel as nib
import random

def ResampleImage_refT2(reference,moving):
# Load Resample Filter
    resample = sitk.ResampleImageFilter()
# Set desired output spacing    
    #out_spacing = list((2.0, 2.0, 2.0))
    
    
#%% Resample T2 to a fixed pixel spacing    
# Get Spacing, Size and PixelIDValue(for padding) of T2    
    ref_original_spacing = reference.GetSpacing()
    ref_original_size =  reference.GetSize()



#%% Resample ADC (or DWI) to a fixed pixel spacing, origin and direction of T2      
    if moving is not None:
        # x,y,z=ref_out_size[::-1]
        xr,yr,zr = ref_original_size[::-1] #Update
        
        mov_original_spacing = moving.GetSpacing()
        mov_original_size =  moving.GetSize()
        mov_pad_value = moving.GetPixelIDValue()
        
        mov_out_size = [
                   int(np.round(
                       size * (spacing_in / spacing_out)
                   ))
                   for size, spacing_in, spacing_out in zip(mov_original_size, mov_original_spacing, ref_original_spacing)
               ]
   
        resample.SetOutputSpacing(ref_original_spacing)
        resample.SetOutputDirection(reference.GetDirection())
        resample.SetOutputOrigin(reference.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetSize(mov_out_size)
        resample.SetDefaultPixelValue(mov_pad_value)
        resample.SetInterpolator(sitk.sitkBSpline)
        mov = resample.Execute(moving)
        mov= sitk.GetArrayFromImage(mov)
        #Update
        pad_mov = np.zeros((xr,yr,zr))
        xm,ym,zm=mov.shape
        #Padding or cropping Image
        x,y,z = np.min(np.vstack(((xr,yr,zr),(xm,ym,zm))),axis=0)
        pad_mov[:x,:y,:z]=mov[:x,:y,:z]
        mov=pad_mov
        ref = sitk.GetArrayFromImage(reference)
    else:
        mov=None

    return ref,mov


def image_mask_resampler(image,mask):

    image = sitk.Cast(image, sitk.sitkUInt32)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    corrected_mask = resampler.Execute(mask)

    return corrected_mask