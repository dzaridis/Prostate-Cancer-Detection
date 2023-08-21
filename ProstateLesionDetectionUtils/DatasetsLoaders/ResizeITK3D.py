import SimpleITK as sitk
import numpy as np

def loadLPS(file_path,return_array=False):
    ITK_image=sitk.ReadImage(file_path)
    ITK_image=sitk.DICOMOrient(ITK_image, 'LPS')
    if return_array:
        Array_image=sitk.GetArrayFromImage(ITK_image)
        return Array_image
    else:
        return ITK_image

def resample_space_size(
    image,
    out_size= (256,256,24),
    out_spacing= (0.5,0.5,3.0),
    is_label = False,
    pad_value = None):
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size


    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(image.GetSize(), image.GetSpacing(), out_spacing)
        ]
    print("OutSize:", out_size)
    print("InSize:", image.GetSize())
    print("InSpacing:", image.GetSpacing())
    print("OutSpacing:", out_spacing)
    print("--------------------------")
    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image