from dataprocess.utils import DicomSeriesReader, GetLargestConnectedCompont, GetLargestConnectedCompontBoundingbox, \
    file_name_path, MorphologicalOperation
import SimpleITK as sitk
import os
import random
import numpy as np

dose1_2 = '1-2 dose'
dose1_4 = '1-4 dose'
dose1_10 = '1-10 dose'
dose1_20 = '1-20 dose'
dose1_50 = '1-50 dose'
dose1_100 = '1-100 dose'
dose1_22 = 'D2'
dose1_42 = 'D4'
dose1_102 = 'D10'
dose1_202 = 'D20'
dose1_502 = 'D50'
dose1_1002 = 'D100'
dose1_23 = 'DRF_2'
dose1_43 = 'DRF_4'
dose1_103 = 'DRF_10'
dose1_203 = 'DRF_20'
dose1_503 = 'DRF_50'
dose1_1003 = 'DRF_100'
dose_full = 'Full_dose'
dose_full2 = 'normal'
dose_full3 = 'NORMAL'

image_dir = "Image"
mask_dir = "Mask"
dose1_2_image = '1-2 dose.nii.gz'
dose1_4_image = '1-4 dose.nii.gz'
dose1_10_image = '1-10 dose.nii.gz'
dose1_20_image = '1-20 dose.nii.gz'
dose1_50_image = '1-50 dose.nii.gz'
dose1_100_image = '1-100 dose.nii.gz'
dose_full_image = 'Full_dose.nii.gz'


def GetROIDataU(flag=1):
    src_data_path = r'D:\challenge\data\2022Ultra-low dose PET\uExplorer'
    out_data_roi_path = r'D:\challenge\data\2022Ultra-low dose PET\ROIProcess'
    data_dirs_path = file_name_path(src_data_path)
    dir_file = 0
    for dir in range(0, len(data_dirs_path)):
        # number58，sub_dirs: ['2.886 x 600 WB D2', '2.886 x 600 WB DRF_10', '2.886 x 600 WB DRF_100',
        # '2.886 x 600 WB DRF_20', '2.886 x 600 WB DRF_4', '2.886 x 600 WB DRF_50', '2.886 x 600 WB normal']
        data_dir_path = src_data_path + '/' + data_dirs_path[dir]
        image_dirs_path = file_name_path(data_dir_path)
        # 读取全部序列图像
        dose1_2_name = image_dirs_path[2]
        dose1_4_name = image_dirs_path[4]
        dose1_10_name = image_dirs_path[0]
        dose1_20_name = image_dirs_path[3]
        dose1_50_name = image_dirs_path[5]
        dose1_100_name = image_dirs_path[1]
        dosefull_name = image_dirs_path[6]
        dose1_2_image_dir = data_dir_path + '/' + dose1_2_name
        dose1_4_image_dir = data_dir_path + '/' + dose1_4_name
        dose1_10_image_dir = data_dir_path + '/' + dose1_10_name
        dose1_20_image_dir = data_dir_path + '/' + dose1_20_name
        dose1_50_image_dir = data_dir_path + '/' + dose1_50_name
        dose1_100_image_dir = data_dir_path + '/' + dose1_100_name
        dosefull_image_dir = data_dir_path + '/' + dosefull_name

        dose1_2_image, _ = DicomSeriesReader(dose1_2_image_dir)
        dose1_4_image, _ = DicomSeriesReader(dose1_4_image_dir)
        dose1_10_image, _ = DicomSeriesReader(dose1_10_image_dir)
        dose1_20_image, _ = DicomSeriesReader(dose1_20_image_dir)
        dose1_50_image, _ = DicomSeriesReader(dose1_50_image_dir)
        dose1_100_image, _ = DicomSeriesReader(dose1_100_image_dir)
        dosefull_image, _ = DicomSeriesReader(dosefull_image_dir)
        # 根据全剂量pet图像进行ROI分割提取，采用固定阈值（50，最大值）然后采用形态学开操作去除边界多余的噪声，得到boundingbox
        minimumfilter = sitk.MinimumMaximumImageFilter()
        minimumfilter.Execute(dosefull_image)

        binary_src = sitk.BinaryThreshold(dosefull_image, 50, minimumfilter.GetMaximum())
        binary_src = MorphologicalOperation(binary_src, 5)
        binary_src = GetLargestConnectedCompont(binary_src)
        boundingbox = GetLargestConnectedCompontBoundingbox(binary_src)
        # print(boundingbox)  # (x,y,z,xlength,ylength,zlength)
        x1, y1, z1, x2, y2, z2 = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[0] + boundingbox[3], \
                                 boundingbox[1] + boundingbox[4], boundingbox[2] + boundingbox[5]
        dose1_2_array = sitk.GetArrayFromImage(dose1_2_image)
        dose1_4_array = sitk.GetArrayFromImage(dose1_4_image)
        dose1_10_array = sitk.GetArrayFromImage(dose1_10_image)
        dose1_20_array = sitk.GetArrayFromImage(dose1_20_image)
        dose1_50_array = sitk.GetArrayFromImage(dose1_50_image)
        dose1_100_array = sitk.GetArrayFromImage(dose1_100_image)
        dosefull_array = sitk.GetArrayFromImage(dosefull_image)

        out_data_roi_dir = out_data_roi_path + '/' + str(flag) + '_' + str(dir_file)
        if not os.path.exists(out_data_roi_dir):
            os.makedirs(out_data_roi_dir)
        # 根据boundingbox得到ROI区域
        roi_dose1_2_array = dose1_2_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_2_image = sitk.GetImageFromArray(roi_dose1_2_array)
        roi_dose1_2_image.SetSpacing(dose1_2_image.GetSpacing())
        roi_dose1_2_image.SetDirection(dose1_2_image.GetDirection())
        roi_dose1_2_image.SetOrigin(dose1_2_image.GetOrigin())
        sitk.WriteImage(roi_dose1_2_image, out_data_roi_dir + '/' + dose1_2 + '.nii.gz')

        roi_dose1_4_array = dose1_4_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_4_image = sitk.GetImageFromArray(roi_dose1_4_array)
        roi_dose1_4_image.SetSpacing(dose1_4_image.GetSpacing())
        roi_dose1_4_image.SetDirection(dose1_4_image.GetDirection())
        roi_dose1_4_image.SetOrigin(dose1_4_image.GetOrigin())
        sitk.WriteImage(roi_dose1_4_image, out_data_roi_dir + '/' + dose1_4 + '.nii.gz')

        roi_dose1_10_array = dose1_10_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_10_image = sitk.GetImageFromArray(roi_dose1_10_array)
        roi_dose1_10_image.SetSpacing(dose1_10_image.GetSpacing())
        roi_dose1_10_image.SetDirection(dose1_10_image.GetDirection())
        roi_dose1_10_image.SetOrigin(dose1_10_image.GetOrigin())
        sitk.WriteImage(roi_dose1_10_image, out_data_roi_dir + '/' + dose1_10 + '.nii.gz')

        roi_dose1_20_array = dose1_20_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_20_image = sitk.GetImageFromArray(roi_dose1_20_array)
        roi_dose1_20_image.SetSpacing(dose1_20_image.GetSpacing())
        roi_dose1_20_image.SetDirection(dose1_20_image.GetDirection())
        roi_dose1_20_image.SetOrigin(dose1_20_image.GetOrigin())
        sitk.WriteImage(roi_dose1_20_image, out_data_roi_dir + '/' + dose1_20 + '.nii.gz')

        roi_dose1_50_array = dose1_50_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_50_image = sitk.GetImageFromArray(roi_dose1_50_array)
        roi_dose1_50_image.SetSpacing(dose1_50_image.GetSpacing())
        roi_dose1_50_image.SetDirection(dose1_50_image.GetDirection())
        roi_dose1_50_image.SetOrigin(dose1_50_image.GetOrigin())
        sitk.WriteImage(roi_dose1_50_image, out_data_roi_dir + '/' + dose1_50 + '.nii.gz')

        roi_dose1_100_array = dose1_100_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_100_image = sitk.GetImageFromArray(roi_dose1_100_array)
        roi_dose1_100_image.SetSpacing(dose1_100_image.GetSpacing())
        roi_dose1_100_image.SetDirection(dose1_100_image.GetDirection())
        roi_dose1_100_image.SetOrigin(dose1_100_image.GetOrigin())
        sitk.WriteImage(roi_dose1_100_image, out_data_roi_dir + '/' + dose1_100 + '.nii.gz')

        roi_dosefull_array = dosefull_array[z1:z2, y1:y2, x1:x2]
        roi_dosefull_image = sitk.GetImageFromArray(roi_dosefull_array)
        roi_dosefull_image.SetSpacing(dosefull_image.GetSpacing())
        roi_dosefull_image.SetDirection(dosefull_image.GetDirection())
        roi_dosefull_image.SetOrigin(dosefull_image.GetOrigin())
        sitk.WriteImage(roi_dosefull_image, out_data_roi_dir + '/' + dose_full + '.nii.gz')

        dir_file = dir_file + 1


def GetROIDataSimension(flag=0):
    src_data_path = r'D:\challenge\data\2022Ultra-low dose PET\Siemens Vision Quadra'
    out_data_roi_path = r'D:\challenge\data\2022Ultra-low dose PET\ROIProcess'
    data_dirs_path = file_name_path(src_data_path)
    dir_file = 0
    for dir in range(len(data_dirs_path)):
        data_dir_path = src_data_path + '/' + data_dirs_path[dir]
        image_dirs_path = file_name_path(data_dir_path)
        # 目录下全部dicom图像序列图像
        for index in range(len(image_dirs_path)):
            if dose1_2 in image_dirs_path[index]:
                dose1_2_name = image_dirs_path[index]
            if dose1_4 in image_dirs_path[index]:
                dose1_4_name = image_dirs_path[index]
            if dose1_10 in image_dirs_path[index]:
                dose1_10_name = image_dirs_path[index]
            if dose1_20 in image_dirs_path[index]:
                dose1_20_name = image_dirs_path[index]
            if dose1_50 in image_dirs_path[index]:
                dose1_50_name = image_dirs_path[index]
            if dose1_100 in image_dirs_path[index]:
                dose1_100_name = image_dirs_path[index]
            if dose_full in image_dirs_path[index]:
                dosefull_name = image_dirs_path[index]
        # 读取全部序列图像
        dose1_2_image_dir = data_dir_path + '/' + dose1_2_name
        dose1_4_image_dir = data_dir_path + '/' + dose1_4_name
        dose1_10_image_dir = data_dir_path + '/' + dose1_10_name
        dose1_20_image_dir = data_dir_path + '/' + dose1_20_name
        dose1_50_image_dir = data_dir_path + '/' + dose1_50_name
        dose1_100_image_dir = data_dir_path + '/' + dose1_100_name
        dosefull_image_dir = data_dir_path + '/' + dosefull_name

        dose1_2_image, _ = DicomSeriesReader(dose1_2_image_dir)
        dose1_4_image, _ = DicomSeriesReader(dose1_4_image_dir)
        dose1_10_image, _ = DicomSeriesReader(dose1_10_image_dir)
        dose1_20_image, _ = DicomSeriesReader(dose1_20_image_dir)
        dose1_50_image, _ = DicomSeriesReader(dose1_50_image_dir)
        dose1_100_image, _ = DicomSeriesReader(dose1_100_image_dir)
        dosefull_image, _ = DicomSeriesReader(dosefull_image_dir)
        # 根据全剂量pet图像进行ROI分割提取，采用固定阈值（50，最大值）然后采用形态学开操作去除边界多余的噪声，得到boundingbox
        minimumfilter = sitk.MinimumMaximumImageFilter()
        minimumfilter.Execute(dosefull_image)

        binary_src = sitk.BinaryThreshold(dosefull_image, 50, minimumfilter.GetMaximum())
        binary_src = MorphologicalOperation(binary_src, 5)
        binary_src = GetLargestConnectedCompont(binary_src)
        boundingbox = GetLargestConnectedCompontBoundingbox(binary_src)
        # print(boundingbox)  # (x,y,z,xlength,ylength,zlength)
        x1, y1, z1, x2, y2, z2 = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[0] + boundingbox[3], \
                                 boundingbox[1] + boundingbox[4], boundingbox[2] + boundingbox[5]
        dose1_2_array = sitk.GetArrayFromImage(dose1_2_image)
        dose1_4_array = sitk.GetArrayFromImage(dose1_4_image)
        dose1_10_array = sitk.GetArrayFromImage(dose1_10_image)
        dose1_20_array = sitk.GetArrayFromImage(dose1_20_image)
        dose1_50_array = sitk.GetArrayFromImage(dose1_50_image)
        dose1_100_array = sitk.GetArrayFromImage(dose1_100_image)
        dosefull_array = sitk.GetArrayFromImage(dosefull_image)

        out_data_roi_dir = out_data_roi_path + '/' + str(flag) + '_' + str(dir_file)
        if not os.path.exists(out_data_roi_dir):
            os.makedirs(out_data_roi_dir)
        # 根据boundingbox得到ROI区域
        roi_dose1_2_array = dose1_2_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_2_image = sitk.GetImageFromArray(roi_dose1_2_array)
        roi_dose1_2_image.SetSpacing(dose1_2_image.GetSpacing())
        roi_dose1_2_image.SetDirection(dose1_2_image.GetDirection())
        roi_dose1_2_image.SetOrigin(dose1_2_image.GetOrigin())
        sitk.WriteImage(roi_dose1_2_image, out_data_roi_dir + '/' + dose1_2 + '.nii.gz')

        roi_dose1_4_array = dose1_4_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_4_image = sitk.GetImageFromArray(roi_dose1_4_array)
        roi_dose1_4_image.SetSpacing(dose1_4_image.GetSpacing())
        roi_dose1_4_image.SetDirection(dose1_4_image.GetDirection())
        roi_dose1_4_image.SetOrigin(dose1_4_image.GetOrigin())
        sitk.WriteImage(roi_dose1_4_image, out_data_roi_dir + '/' + dose1_4 + '.nii.gz')

        roi_dose1_10_array = dose1_10_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_10_image = sitk.GetImageFromArray(roi_dose1_10_array)
        roi_dose1_10_image.SetSpacing(dose1_10_image.GetSpacing())
        roi_dose1_10_image.SetDirection(dose1_10_image.GetDirection())
        roi_dose1_10_image.SetOrigin(dose1_10_image.GetOrigin())
        sitk.WriteImage(roi_dose1_10_image, out_data_roi_dir + '/' + dose1_10 + '.nii.gz')

        roi_dose1_20_array = dose1_20_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_20_image = sitk.GetImageFromArray(roi_dose1_20_array)
        roi_dose1_20_image.SetSpacing(dose1_20_image.GetSpacing())
        roi_dose1_20_image.SetDirection(dose1_20_image.GetDirection())
        roi_dose1_20_image.SetOrigin(dose1_20_image.GetOrigin())
        sitk.WriteImage(roi_dose1_20_image, out_data_roi_dir + '/' + dose1_20 + '.nii.gz')

        roi_dose1_50_array = dose1_50_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_50_image = sitk.GetImageFromArray(roi_dose1_50_array)
        roi_dose1_50_image.SetSpacing(dose1_50_image.GetSpacing())
        roi_dose1_50_image.SetDirection(dose1_50_image.GetDirection())
        roi_dose1_50_image.SetOrigin(dose1_50_image.GetOrigin())
        sitk.WriteImage(roi_dose1_50_image, out_data_roi_dir + '/' + dose1_50 + '.nii.gz')

        roi_dose1_100_array = dose1_100_array[z1:z2, y1:y2, x1:x2]
        roi_dose1_100_image = sitk.GetImageFromArray(roi_dose1_100_array)
        roi_dose1_100_image.SetSpacing(dose1_100_image.GetSpacing())
        roi_dose1_100_image.SetDirection(dose1_100_image.GetDirection())
        roi_dose1_100_image.SetOrigin(dose1_100_image.GetOrigin())
        sitk.WriteImage(roi_dose1_100_image, out_data_roi_dir + '/' + dose1_100 + '.nii.gz')

        roi_dosefull_array = dosefull_array[z1:z2, y1:y2, x1:x2]
        roi_dosefull_image = sitk.GetImageFromArray(roi_dosefull_array)
        roi_dosefull_image.SetSpacing(dosefull_image.GetSpacing())
        roi_dosefull_image.SetDirection(dosefull_image.GetDirection())
        roi_dosefull_image.SetOrigin(dosefull_image.GetOrigin())
        sitk.WriteImage(roi_dosefull_image, out_data_roi_dir + '/' + dose_full + '.nii.gz')

        dir_file = dir_file + 1


def subpatchgenerate(trainImage, trainMask, petarray, maskarray, subnnumber, shape, subsetindex):
    imagez, imagey, imagex = petarray.shape[0], petarray.shape[1], petarray.shape[2]
    xrange = imagex - shape[0]
    yrange = imagey - shape[1]
    zrange = imagez - shape[2]
    index = 0
    if xrange >= 0 and yrange >= 0 and zrange >= 0:
        while index <= subnnumber:
            x = random.randint(0, xrange)
            y = random.randint(0, yrange)
            z = random.randint(0, zrange)
            petimageroiarray = petarray[z:z + shape[2], y:y + shape[1], x:x + shape[0]]
            maskroiarray = maskarray[z:z + shape[2], y:y + shape[1], x:x + shape[0]]
            filepath1 = trainImage + "\\" + str(subsetindex) + "_" + str(index) + ".npy"
            np.save(filepath1, petimageroiarray)
            filepath = trainMask + "\\" + str(subsetindex) + "_" + str(index) + ".npy"
            np.save(filepath, maskroiarray)
            # sitk.WriteImage(sitk.GetImageFromArray(petimageroiarray), 'pet.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(maskroiarray), 'mask.nii.gz')
            index = index + 1


def subpatchgeneratewithfixedstep(trainImage, trainMask, imagedata, maskdata, fixstep, shape, subsetindex):
    z_size, y_size, x_size = imagedata.shape[0], imagedata.shape[1], imagedata.shape[2]
    xrange = x_size - shape[0]
    yrange = y_size - shape[1]
    zrange = z_size - shape[2]
    index = 0
    if xrange >= 0 and yrange >= 0 and zrange >= 0:
        for z in range(0, z_size, shape[2] // fixstep):
            for y in range(0, y_size, shape[1] // fixstep):
                for x in range(0, x_size, shape[0] // fixstep):
                    x_min = x
                    x_max = x_min + shape[0]
                    if x_max > x_size:
                        x_max = x_size
                        x_min = x_size - shape[0]
                    y_min = y
                    y_max = y_min + shape[1]
                    if y_max > y_size:
                        y_max = y_size
                        y_min = y_size - shape[1]
                    z_min = z
                    z_max = z_min + shape[2]
                    if z_max > z_size:
                        z_max = z_size
                        z_min = z_size - shape[2]
                    resizeimagearray = imagedata[z_min:z_max, y_min:y_max, x_min:x_max]
                    resizemaskarray = maskdata[z_min:z_max, y_min:y_max, x_min:x_max]
                    if np.sum(resizemaskarray) / (shape[0] * shape[1] * shape[2]) > 0.01:
                        # sitk.WriteImage(sitk.GetImageFromArray(resizeimagearray), str(index) + 'roiimage.nii.gz')
                        # sitk.WriteImage(sitk.GetImageFromArray(resizemaskarray), str(index) + 'roimask.nii.gz')
                        filepath1 = trainImage + "\\" + str(subsetindex) + "_" + str(index) + ".npy"
                        filepath = trainMask + "\\" + str(subsetindex) + "_" + str(index) + ".npy"
                        np.save(filepath1, resizeimagearray)
                        np.save(filepath, resizemaskarray)
                        index = index + 1


def preparesampling3dtraindatawithpatch(datapath, pet_image, pet_mask, trainImage, trainMask, subnnumber,
                                        shape=(96, 96, 96)):
    all_files = file_name_path(datapath, True, False)
    for subsetindex in range(len(all_files)):
        traindata_dir = datapath + '/' + all_files[subsetindex]
        traindata_image = traindata_dir + '/' + pet_image
        traindata_mask = traindata_dir + '/' + pet_mask
        sitk_image = sitk.ReadImage(traindata_image)
        sitk_mask = sitk.ReadImage(traindata_mask)
        array_image = sitk.GetArrayFromImage(sitk_image)
        array_mask = sitk.GetArrayFromImage(sitk_mask)
        subpatchgenerate(trainImage, trainMask, array_image, array_mask, subnnumber, shape, subsetindex)


def preparetraindata():
    """
    :return:
    """
    src_train_path = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\train"
    source_process_path = r"E:\challenge\data\2022Ultra-low dose PET\task_dose1_2\train"
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    if not os.path.exists(outputimagepath):
        os.makedirs(outputimagepath)
    if not os.path.exists(outputlabelpath):
        os.makedirs(outputlabelpath)
    preparesampling3dtraindatawithpatch(src_train_path, dose1_2_image, dose_full_image, outputimagepath,
                                        outputlabelpath, 15, (160, 96, 256))


def preparevalidationdata():
    """
    :return:
    """
    src_train_path = r"D:\challenge\data\2022Ultra-low dose PET\ROIProcess\validation"
    source_process_path = r"E:\challenge\data\2022Ultra-low dose PET\task_dose1_2\validation"
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    if not os.path.exists(outputimagepath):
        os.makedirs(outputimagepath)
    if not os.path.exists(outputlabelpath):
        os.makedirs(outputlabelpath)
    preparesampling3dtraindatawithpatch(src_train_path, dose1_2_image, dose_full_image, outputimagepath,
                                        outputlabelpath, 15, (160, 96, 256))


if __name__ == "__main__":
    # GetROIDataSimension()
    # GetROIDataU()
    preparetraindata()
    preparevalidationdata()
