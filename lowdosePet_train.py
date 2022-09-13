import pandas as pd
import torch
import os
from model import *
import numpy as np

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def trainbinaryunet3dmse2():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_2.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_2.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_2/mse',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dl12():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_2.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_2.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='L1')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_2/L1',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dmse4():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_4.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_4.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_4/mse',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dl14():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_4.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_4.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='L1')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_4/L1',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dmse10():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_10.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_10.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_10/mse',
                        epochs=50, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dl110():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_10.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_10.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='L1')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_10/L1',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dmse20():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_20.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_20.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_20/mse',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dl120():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_20.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_20.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='L1')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_20/L1',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dmse50():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_50.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_50.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_50/mse',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dl150():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_50.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_50.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='L1')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_50/L1',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dmse100():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_100.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_100.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='MSE')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_100/mse',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


def trainbinaryunet3dl1100():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data\\lowdosePet\\traindose1_100.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = 'dataprocess\\data\\lowdosePet\\validatadose1_100.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = UNet3dRegressionModel(image_depth=256, image_height=96, image_width=160, image_channel=1, numclass=1,
                                   batch_size=1, loss_name='L1')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/lowdosePet/dose1_100/L1',
                        epochs=20, showwind=[16, 16])
    unet3d.clear_GPU_cache()


if __name__ == '__main__':
    trainbinaryunet3dmse2()
    trainbinaryunet3dl12()
    # trainbinaryunet3dmse4()
    # trainbinaryunet3dl14()
    # trainbinaryunet3dmse10()
    # trainbinaryunet3dl110()
    # trainbinaryunet3dmse20()
    # trainbinaryunet3dl120()
    # trainbinaryunet3dmse50()
    # trainbinaryunet3dl150()
    # trainbinaryunet3dmse100()
    # trainbinaryunet3dl1100()
