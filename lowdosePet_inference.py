import torch
import os
from model import *
from dataprocess.utils import file_name_path
import SimpleITK as sitk

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()

dose1_2_image = '1-2 dose.nii.gz'
dose1_4_image = '1-4 dose.nii.gz'
dose1_10_image = '1-10 dose.nii.gz'
dose1_20_image = '1-20 dose.nii.gz'
dose1_50_image = '1-50 dose.nii.gz'
dose1_100_image = '1-100 dose.nii.gz'


def inferencebinaryunet3d2():
    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE', inference=True,
                                   model_path=r"log\lowdosePet\dose1_2\L1\unet_19.pth")
    datapath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validation"
    outputpath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validaitonpd"
    image_name = dose1_2_image
    image_path_list = file_name_path(datapath, True, False)
    for i in range(len(image_path_list)):
        filedirpath = datapath + '/' + image_path_list[i]
        petimage_file = filedirpath + "/" + image_name
        petsrc = sitk.ReadImage(petimage_file)
        sitk_mask = unet3d.inference_patch(petsrc)
        maskpath = outputpath + '/' + image_path_list[i]
        if not os.path.exists(maskpath):
            os.makedirs(maskpath)
        maskpathname = maskpath + "/L1" + image_name
        sitk.WriteImage(sitk_mask, maskpathname)


def inferencebinaryunet3d4():
    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE', inference=True,
                                   model_path=r"log\lowdosePet\dose1_4\L1\unet_19.pth")
    datapath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validation"
    outputpath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validaitonpd"
    image_name = dose1_4_image
    image_path_list = file_name_path(datapath, True, False)
    for i in range(len(image_path_list)):
        filedirpath = datapath + '/' + image_path_list[i]
        petimage_file = filedirpath + "/" + image_name
        petsrc = sitk.ReadImage(petimage_file)
        sitk_mask = unet3d.inference_patch(petsrc)
        maskpath = outputpath + '/' + image_path_list[i]
        if not os.path.exists(maskpath):
            os.makedirs(maskpath)
        maskpathname = maskpath + "/mse" + image_name
        sitk.WriteImage(sitk_mask, maskpathname)


def inferencebinaryunet3d10():
    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE', inference=True,
                                   model_path=r"log\lowdosePet\dose1_10\L1\unet_19.pth")
    datapath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validation"
    outputpath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validaitonpd"
    image_name = dose1_10_image
    image_path_list = file_name_path(datapath, True, False)
    for i in range(len(image_path_list)):
        filedirpath = datapath + '/' + image_path_list[i]
        petimage_file = filedirpath + "/" + image_name
        petsrc = sitk.ReadImage(petimage_file)
        sitk_mask = unet3d.inference_patch(petsrc)
        maskpath = outputpath + '/' + image_path_list[i]
        if not os.path.exists(maskpath):
            os.makedirs(maskpath)
        maskpathname = maskpath + "/L1" + image_name
        sitk.WriteImage(sitk_mask, maskpathname)


def inferencebinaryunet3d20():
    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE', inference=True,
                                   model_path=r"log\lowdosePet\dose1_20\L1\unet_19.pth")
    datapath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validation"
    outputpath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validaitonpd"
    image_name = dose1_20_image
    image_path_list = file_name_path(datapath, True, False)
    for i in range(len(image_path_list)):
        filedirpath = datapath + '/' + image_path_list[i]
        petimage_file = filedirpath + "/" + image_name
        petsrc = sitk.ReadImage(petimage_file)
        sitk_mask = unet3d.inference_patch(petsrc)
        maskpath = outputpath + '/' + image_path_list[i]
        if not os.path.exists(maskpath):
            os.makedirs(maskpath)
        maskpathname = maskpath + "/mse" + image_name
        sitk.WriteImage(sitk_mask, maskpathname)


def inferencebinaryunet3d50():
    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE', inference=True,
                                   model_path=r"log\lowdosePet\dose1_50\L1\unet_19.pth")
    datapath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validation"
    outputpath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validaitonpd"
    image_name = dose1_50_image
    image_path_list = file_name_path(datapath, True, False)
    for i in range(len(image_path_list)):
        filedirpath = datapath + '/' + image_path_list[i]
        petimage_file = filedirpath + "/" + image_name
        petsrc = sitk.ReadImage(petimage_file)
        sitk_mask = unet3d.inference_patch(petsrc)
        maskpath = outputpath + '/' + image_path_list[i]
        if not os.path.exists(maskpath):
            os.makedirs(maskpath)
        maskpathname = maskpath + "/L1" + image_name
        sitk.WriteImage(sitk_mask, maskpathname)


def inferencebinaryunet3d100():
    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE', inference=True,
                                   model_path=r"log\lowdosePet\dose1_100\L1\unet_19.pth")
    datapath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validation"
    outputpath = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validaitonpd"
    image_name = dose1_100_image
    image_path_list = file_name_path(datapath, True, False)
    for i in range(len(image_path_list)):
        filedirpath = datapath + '/' + image_path_list[i]
        petimage_file = filedirpath + "/" + image_name
        petsrc = sitk.ReadImage(petimage_file)
        sitk_mask = unet3d.inference_patch(petsrc)
        maskpath = outputpath + '/' + image_path_list[i]
        if not os.path.exists(maskpath):
            os.makedirs(maskpath)
        maskpathname = maskpath + "/mse" + image_name
        sitk.WriteImage(sitk_mask, maskpathname)


if __name__ == '__main__':
    inferencebinaryunet3d2()
    inferencebinaryunet3d4()
    inferencebinaryunet3d10()
    inferencebinaryunet3d20()
    inferencebinaryunet3d50()
    inferencebinaryunet3d100()
