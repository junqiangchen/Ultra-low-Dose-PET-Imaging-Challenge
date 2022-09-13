import numpy as np
import torch
import os
from model import *
from dataprocess.utils import file_name_path, MorphologicalOperation, GetLargestConnectedCompont, \
    GetLargestConnectedCompontBoundingbox
import SimpleITK as sitk

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()

dose1_2_image = 'drf_2.nii.gz'
dose1_4_image = 'drf_4.nii.gz'
dose1_10_image = 'drf_10.nii.gz'
dose1_20_image = 'drf_20.nii.gz'
dose1_50_image = 'drf_50.nii.gz'
dose1_100_image = 'drf_100.nii.gz'


def inferencebinaryunet3d():
    datapath = r'D:\challenge\data\2022Ultra-low dose PET\Test\test'
    outputpath = r'D:\challenge\data\2022Ultra-low dose PET\Test\test_mask'
    outputimagepath = r'D:\challenge\data\2022Ultra-low dose PET\Test\test_image'
    datadirs = file_name_path(datapath, True, False)
    for index in range(len(datadirs)):
        datadirpath = datapath + '/' + datadirs[index]
        imagefiles = file_name_path(datadirpath, False, True)
        for i in range(len(imagefiles)):
            imagefilename = imagefiles[i]
            imagefilepath = datadirpath + '/' + imagefilename
            dose_image = sitk.ReadImage(imagefilepath)

            minimumfilter = sitk.MinimumMaximumImageFilter()
            minimumfilter.Execute(dose_image)
            binary_src = sitk.BinaryThreshold(dose_image, 50, minimumfilter.GetMaximum())
            binary_src = MorphologicalOperation(binary_src, 5)
            binary_src = GetLargestConnectedCompont(binary_src)
            boundingbox = GetLargestConnectedCompontBoundingbox(binary_src)
            # print(boundingbox)  # (x,y,z,xlength,ylength,zlength)
            x1, y1, z1, x2, y2, z2 = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[0] + boundingbox[3], \
                                     boundingbox[1] + boundingbox[4], boundingbox[2] + boundingbox[5]
            dose_image_array = sitk.GetArrayFromImage(dose_image)
            roi_dose_image_array = dose_image_array[z1:z2, y1:y2, x1:x2]
            roi_dose_image_sitk = sitk.GetImageFromArray(roi_dose_image_array)
            roi_dose_image_sitk.SetSpacing(dose_image.GetSpacing())
            roi_dose_image_sitk.SetDirection(dose_image.GetDirection())
            roi_dose_image_sitk.SetOrigin(dose_image.GetOrigin())

            if dose1_2_image == imagefilename:
                model_path = r"log\lowdosePet\dose1_2\L1\unet_19.pth"
            if dose1_4_image == imagefilename:
                model_path = r"log\lowdosePet\dose1_4\L1\unet_19.pth"
            if dose1_10_image == imagefilename:
                model_path = r"log\lowdosePet\dose1_10\L1\unet_19.pth"
            if dose1_20_image == imagefilename:
                model_path = r"log\lowdosePet\dose1_20\L1\unet_19.pth"
            if dose1_50_image == imagefilename:
                model_path = r"log\lowdosePet\dose1_50\L1\unet_19.pth"
            if dose1_100_image == imagefilename:
                model_path = r"log\lowdosePet\dose1_100\L1\unet_19.pth"
            unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1,
                                           numclass=1, batch_size=1, loss_name='L1', inference=True,
                                           model_path=model_path)
            sitk_mask_predict = unet3d.inference_patch(roi_dose_image_sitk)
            unet3d.clear_GPU_cache()
            final_mask_predict = np.zeros_like(dose_image_array)
            array_mask_predict = sitk.GetArrayFromImage(sitk_mask_predict)
            final_mask_predict[z1:z2, y1:y2, x1:x2] = array_mask_predict.copy()
            final_sitk_mask = sitk.GetImageFromArray(final_mask_predict)
            final_sitk_mask.SetOrigin(dose_image.GetOrigin())
            final_sitk_mask.SetSpacing(dose_image.GetSpacing())
            final_sitk_mask.SetDirection(dose_image.GetDirection())
            maskpath = outputpath + '/' + datadirs[index] + '.nii.gz'
            sitk.WriteImage(final_sitk_mask, maskpath)
            maskpath1 = outputimagepath + '/' + datadirs[index] + '.nii.gz'
            sitk.WriteImage(dose_image, maskpath1)


if __name__ == '__main__':
    inferencebinaryunet3d()
