import torch
from networks.Unet2d import UNet2d
from networks.Unet3d import UNet3d
from networks import initialize_weights
from .dataset import datasetModelSegwithopencv, datasetModelSegwithnpy, datasetModelRegressionwithopencv, \
    datasetModelRegressionwithnpy
from torch.utils.data import DataLoader
from .losses import BinaryDiceLoss, BinaryFocalLoss, BinaryCrossEntropyLoss, BinaryCrossEntropyDiceLoss, \
    MutilDiceLoss, MutilFocalLoss, MutilCrossEntropyLoss, L1Loss, MSELoss
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from .metric import dice_coeff, iou_coeff, multiclass_dice_coeff, multiclass_iou_coeff, calc_psnr, calc_mse
from .visualization import plot_result, save_images2d, save_images3d, save_images2dregression, save_images3dregression
from pathlib import Path
import time
import os
import cv2
from dataprocess.utils import resize_image_itkwithsize, ConvertitkTrunctedValue, normalize, resize_image_itk
import SimpleITK as sitk
import multiprocessing
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


class BinaryUNet2dModel(object):
    """
    Unet2d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, loss_name='BinaryDiceLoss',
                 inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = 0.25
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = UNet2d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        # Number of workers
        dataset = datasetModelSegwithopencv(images, labels,
                                            targetsize=(self.image_channel, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'BinaryCrossEntropyLoss':
            return BinaryCrossEntropyLoss()
        if lossname is 'BinaryDiceLoss':
            return BinaryDiceLoss()
        if lossname is 'BinaryCrossEntropyDiceLoss':
            return BinaryCrossEntropyDiceLoss()
        if lossname is 'BinaryFocalLoss':
            return BinaryFocalLoss()
        if lossname is 'MSE':
            return MSELoss()
        if lossname is 'L1':
            return L1Loss()

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname is 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "BinaryUNet2d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_height, self.image_width))
        print(self.model)
        showpixelvalue = 255.
        if self.numclass > 1:
            showpixelvalue = showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.AdamW(self.model.parameters(), lr=lr)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask, True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            trainshow = True
            for batch in train_loader:
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                y[y != 0] = 1
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
                loss = lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                if trainshow:
                    # save_images
                    savepath = model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                    save_images2d(pred[0], y[0], savepath, pixelvalue=showpixelvalue)
                    trainshow = False
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss.cpu().detach().numpy())
                totalTrainAccu.append(accu.cpu().detach().numpy())
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            with torch.no_grad():
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    y[y != 0] = 1
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit, pred = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    if trainshow:
                        # save_images
                        savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                        save_images2d(pred[0], y[0], savepath, pixelvalue=showpixelvalue)
                        trainshow = False
                    totalValidationLoss.append(loss.cpu().detach().numpy())
                    totalValiadtionAccu.append(accu.cpu().detach().numpy())
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # lr_scheduler.step(avgValidationLoss)
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss)
            H["valdation_loss"].append(avgValidationLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 1
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        imageresize = (imageresize - imageresize.mean()) / imageresize.std()
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        out_mask = self.predict(imageresize)
        # resize mask to src image size
        out_mask = cv2.resize(out_mask, image.shape, interpolation=cv2.INTER_NEAREST)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class MutilUNet2dModel(object):
    """
    Unet2d with mutil class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, loss_name='MutilFocalLoss',
                 inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = [1.] * self.numclass
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = UNet2d(self.image_channel, self.numclass)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)
        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithopencv(images, labels,
                                            targetsize=(self.image_channel, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'MutilCrossEntropyLoss':
            return MutilCrossEntropyLoss(alpha=self.alpha)
        if lossname is 'MutilDiceLoss':
            return MutilDiceLoss(alpha=self.alpha)
        if lossname is 'MutilFocalLoss':
            return MutilFocalLoss(alpha=self.alpha, gamma=self.gamma)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname is 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "MutilUNet2d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_height, self.image_width))
        print(self.model)
        showpixelvalue = 255.
        if self.numclass > 1:
            showpixelvalue = showpixelvalue // (self.numclass - 1)
        # 1、initialize net weight init loss function and optimizer
        self.model.apply(initialize_weights)
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.AdamW(self.model.parameters(), lr=lr)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask, True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            trainshow = True
            for batch in train_loader:
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
                loss = lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                if trainshow:
                    savepath = model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                    save_images2d(torch.argmax(pred[0], 0), y[0], savepath, pixelvalue=showpixelvalue)
                    trainshow = False
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss.cpu().detach().numpy())
                totalTrainAccu.append(accu.cpu().detach().numpy())
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            with torch.no_grad():
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit, pred = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    if trainshow:
                        # save_images
                        savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                        save_images2d(torch.argmax(pred[0], 0), y[0], savepath, pixelvalue=showpixelvalue)
                        trainshow = False
                    totalValidationLoss.append(loss.cpu().detach().numpy())
                    totalValiadtionAccu.append(accu.cpu().detach().numpy())
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss)
            H["valdation_loss"].append(avgValidationLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 1
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        imageresize = (imageresize - imageresize.mean()) / imageresize.std()
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        out_mask = self.predict(imageresize)
        # resize mask to src image size
        out_mask = cv2.resize(out_mask, image.shape, interpolation=cv2.INTER_NEAREST)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class BinaryUNet3dModel(object):
    """
    Unet3d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='BinaryDiceLoss', inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = 0.25
        self.gamma = 2

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = UNet3d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithnpy(images, labels,
                                         targetsize=(
                                             self.image_channel, self.image_depth, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'BinaryCrossEntropyLoss':
            return BinaryCrossEntropyLoss()
        if lossname is 'BinaryDiceLoss':
            return BinaryDiceLoss()
        if lossname is 'BinaryCrossEntropyDiceLoss':
            return BinaryCrossEntropyDiceLoss()
        if lossname is 'BinaryFocalLoss':
            return BinaryFocalLoss()

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname is 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3,
                     showwind=[8, 8]):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "BinaryUNet3d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_depth, self.image_height, self.image_width))
        print(self.model)
        showpixelvalue = 255.
        if self.numclass > 1:
            showpixelvalue = showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.AdamW(self.model.parameters(), lr=lr)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask, True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            trainshow = True
            # 4.3、loop over the training set
            for batch in train_loader:
                # x should tensor with shape (N,C,D,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,D,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                y[y != 0] = 1
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
                loss = lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                # save_images
                if trainshow:
                    images = x[0].detach().cpu().squeeze().numpy()
                    sitk_image = sitk.GetImageFromArray(images)
                    sitk.WriteImage(sitk_image, model_dir + "/trainimage.nii.gz")
                    savepath = model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                    save_images3d(pred[0], y[0], showwind, savepath, pixelvalue=showpixelvalue)
                    trainshow = False
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss.cpu().detach().numpy())
                totalTrainAccu.append(accu.cpu().detach().numpy())
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            with torch.no_grad():
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    y[y != 0] = 1
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit, pred = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    if trainshow:
                        # save_images
                        images = x[0].detach().cpu().squeeze().numpy()
                        sitk_image = sitk.GetImageFromArray(images)
                        sitk.WriteImage(sitk_image, model_dir + "/validimage.nii.gz")
                        savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                        save_images3d(pred[0], y[0], showwind, savepath, pixelvalue=showpixelvalue)
                        trainshow = False
                    totalValidationLoss.append(loss.cpu().detach().numpy())
                    totalValiadtionAccu.append(accu.cpu().detach().numpy())
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # lr_scheduler.step(avgValidationLoss)
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss)
            H["valdation_loss"].append(avgValidationLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 1
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, imagesitk, newSize=(96, 96, 96)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(), sitk.sitkLinear)
        resizeimagesitk = ConvertitkTrunctedValue(resizeimagesitk, 100, 0, 'meanstd')
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        # imageresize = normalize(imageresize)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        out_mask = self.predict(imageresize)
        # resize mask to src image size,should rewrite
        out_mask_sitk = sitk.GetImageFromArray(out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, final_out_mask_sitk = resize_image_itkwithsize(out_mask_sitk, imagesitk.GetSize(), newSize,
                                                          sitk.sitkNearestNeighbor)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def inference_patch(self, imagesitk, newSpacing=(0.5, 0.5, 0.5)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itk(imagesitk, newSpacing, imagesitk.GetSpacing(), sitk.sitkLinear)
        resizeimagesitk = ConvertitkTrunctedValue(resizeimagesitk, -800, -1024, 'meanstd')
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        # predict patch
        stepx = self.image_width // 2
        stepy = self.image_height // 2
        stepz = self.image_depth // 2
        out_mask = np.zeros((D, H, W))
        for z in range(0, D, stepz):
            for y in range(0, H, stepy):
                for x in range(0, W, stepx):
                    x_min = x
                    x_max = x_min + self.image_width
                    if x_max > W:
                        x_max = W
                        x_min = W - self.image_width
                    y_min = y
                    y_max = y_min + self.image_height
                    if y_max > H:
                        y_max = H
                        y_min = H - self.image_height
                    z_min = z
                    z_max = z_min + self.image_depth
                    if z_max > D:
                        z_max = D
                        z_min = D - self.image_depth
                    patch_xs = imageresize[:, z_min:z_max, y_min:y_max, x_min:x_max]
                    predictresult = self.predict(patch_xs)
                    out_mask[z_min:z_max, y_min:y_max, x_min:x_max] = out_mask[z_min:z_max, y_min:y_max, x_min:x_max] \
                                                                      + predictresult.copy()
        # resize mask to src image size,should rewrite
        out_mask[out_mask != 0] = 1
        out_mask_sitk = sitk.GetImageFromArray(out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, resize_out_mask_sitk = resize_image_itk(out_mask_sitk, imagesitk.GetSpacing(), newSpacing,
                                                   sitk.sitkNearestNeighbor)
        _, final_out_mask_sitk = resize_image_itkwithsize(resize_out_mask_sitk, imagesitk.GetSize(),
                                                          resize_out_mask_sitk.GetSize(), sitk.sitkNearestNeighbor)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class MutilUNet3dModel(object):
    """
    UNet3d with mutil class,should rewrite the dataset class
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='MutilFocalLoss', inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'dice'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass

        self.alpha = [1.] * self.numclass
        # self.alpha = [1., 5., 1., 5., 3.]
        self.gamma = 3

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = UNet3d(self.image_channel, self.numclass)
        self.model.to(device=self.device)
        self.alpha = torch.as_tensor(self.alpha).contiguous().to(self.device)
        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelSegwithnpy(images, labels,
                                         targetsize=(
                                             self.image_channel, self.image_depth, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'MutilCrossEntropyLoss':
            return MutilCrossEntropyLoss(alpha=self.alpha)
        if lossname is 'MutilDiceLoss':
            return MutilDiceLoss(alpha=self.alpha)
        if lossname is 'MutilFocalLoss':
            return MutilFocalLoss(alpha=self.alpha, gamma=self.gamma)

    def _accuracy_function(self, accuracyname, input, target):
        if accuracyname is 'dice':
            if self.numclass == 1:
                return dice_coeff(input, target)
            else:
                return multiclass_dice_coeff(input, target)
        if accuracyname is 'iou':
            if self.numclass == 1:
                return iou_coeff(input, target)
            else:
                return multiclass_iou_coeff(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3,
                     showwind=[8, 8]):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        MODEL_PATH = os.path.join(model_dir, "MutilUNet3d.pth")
        summary(self.model, input_size=(self.image_channel, self.image_depth, self.image_height, self.image_width))
        print(self.model)
        showpixelvalue = 255.
        if self.numclass > 1:
            showpixelvalue = showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.AdamW(self.model.parameters(), lr=lr)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask, True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        best_validation_dsc = 0.0
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            trainshow = True
            # 4.3、loop over the training set
            for batch in train_loader:
                # x should tensor with shape (N,C,D,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,D,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit, pred = self.model(x)
                loss = lossFunc(pred_logit, y)
                accu = self._accuracy_function(self.accuracyname, pred, y)
                if trainshow:
                    savepath = model_dir + '/' + str(e + 1) + "_train_EPOCH_"
                    save_images3d(torch.argmax(pred[0], 0), y[0], showwind, savepath,
                                  pixelvalue=showpixelvalue)
                    trainshow = False
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss.cpu().detach().numpy())
                totalTrainAccu.append(accu.cpu().detach().numpy())
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            with torch.no_grad():
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit, pred = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    accu = self._accuracy_function(self.accuracyname, pred, y)
                    if trainshow:
                        # save_images
                        savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                        save_images3d(torch.argmax(pred[0], 0), y[0], showwind, savepath,
                                      pixelvalue=showpixelvalue)
                        trainshow = False
                    totalValidationLoss.append(loss.cpu().detach().numpy())
                    totalValiadtionAccu.append(accu.cpu().detach().numpy())
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss)
            H["valdation_loss"].append(avgValidationLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            if avgValidationAccu > best_validation_dsc:
                best_validation_dsc = avgValidationAccu
                # best_model_params = self.model.state_dict()
                # serialize best model to disk
                torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img, out_threshold=0.5):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            _, output = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
        # 5、get numpy result
        if self.numclass == 1:
            out_mask = (full_mask_np > out_threshold)
            out_mask = out_mask * 1
        else:
            out_mask = np.argmax(full_mask_np, axis=0)
            out_mask = np.squeeze(out_mask)
        return out_mask.astype(np.uint8)

    def inference(self, imagesitk, newSize=(96, 96, 96)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(), sitk.sitkLinear)
        # resizeimagesitk = ConvertitkTrunctedValue(resizeimagesitk, 100, -100, 'meanstd')
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        imageresize = normalize(imageresize)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        out_mask = self.predict(imageresize)
        # resize mask to src image size,should rewrite
        out_mask_sitk = sitk.GetImageFromArray(out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, final_out_mask_sitk = resize_image_itkwithsize(out_mask_sitk, imagesitk.GetSize(), newSize,
                                                          sitk.sitkNearestNeighbor)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class UNet2dRegressionModel(object):
    """
    Unet2d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, loss_name='MSE',
                 inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'PSNR'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = UNet2d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        # Number of workers
        dataset = datasetModelRegressionwithopencv(images, labels,
                                                   targetsize=(self.image_channel, self.image_height, self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'MSE':
            return MSELoss()
        if lossname is 'L1':
            return L1Loss()

    def _accuracy_function(self, accuracyname, input, target, mean, std):
        if accuracyname is 'PSNR':
            return calc_psnr(input, target, mean, std)
        if accuracyname is 'MSE':
            return calc_mse(input, target)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        summary(self.model, input_size=(self.image_channel, self.image_height, self.image_width))
        print(self.model)
        showpixelvalue = 1.
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.AdamW(self.model.parameters(), lr=lr)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask, True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            trainshow = True
            for batch in train_loader:
                # x should tensor with shape (N,C,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                mean = batch['mean']
                std = batch['std']
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss nad accu
                pred_logit, _ = self.model(x)
                loss = lossFunc(pred_logit, y)
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                accu = self._accuracy_function(self.accuracyname, pred_logit, y, mean, std)
                if trainshow:
                    # save_images
                    savepath = model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                    save_images2dregression(x[0] * std[0] + mean[0],
                                            pred_logit[0] * std[0] + mean[0],
                                            y[0] * std[0] + mean[0],
                                            savepath, pixelvalue=showpixelvalue)
                    trainshow = False
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss.cpu().detach().numpy())
                totalTrainAccu.append(accu.cpu().detach().numpy())
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            with torch.no_grad():
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    mean = batch['mean']
                    std = batch['std']
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit, _ = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    accu = self._accuracy_function(self.accuracyname, pred_logit, y, mean, std)
                    # save_images
                    if trainshow:
                        savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                        save_images2dregression(x[0] * std[0] + mean[0],
                                                pred_logit[0] * std[0] + mean[0],
                                                y[0] * std[0] + mean[0],
                                                savepath, pixelvalue=showpixelvalue)
                        trainshow = False
                    totalValidationLoss.append(loss.cpu().detach().numpy())
                    totalValiadtionAccu.append(accu.cpu().detach().numpy())
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # lr_scheduler.step(avgValidationLoss)
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss)
            H["valdation_loss"].append(avgValidationLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            # serialize best model to disk
            MODEL_PATH = os.path.join(model_dir, "unet_%d.pth" % e)
            torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            pred_logit, _ = self.model(img)
            probs = pred_logit[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
            return full_mask_np

    def inference(self, image):
        # resize image and normalization
        imageresize = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        mean = imageresize.mean()
        std = imageresize.std()
        imageresize = (imageresize - mean) / std
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        out_mask = self.predict(imageresize)
        out_mask = out_mask * std + mean
        out_mask = np.clip(out_mask, 0, 255).astype('uint8')
        # resize mask to src image size
        out_mask = cv2.resize(out_mask, image.shape, interpolation=cv2.INTER_LINEAR)
        return out_mask

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())


class UNet3dRegressionModel(object):
    """
    Unet3d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_depth, image_height, image_width, image_channel, numclass, batch_size,
                 loss_name='MSE', inference=False, model_path=None, use_cuda=True):
        self.batch_size = batch_size
        self.loss_name = loss_name
        self.accuracyname = 'PSNR'
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.image_channel = image_channel
        self.numclass = numclass

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = UNet3d(self.image_channel, self.numclass)
        self.model.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelRegressionwithnpy(images, labels,
                                                targetsize=(self.image_channel, self.image_depth, self.image_height,
                                                            self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _loss_function(self, lossname):
        if lossname is 'MSE':
            return MSELoss()
        if lossname is 'L1':
            return L1Loss()

    def _accuracy_function(self, accuracyname, input, target, mean, std):
        if accuracyname is 'PSNR':
            return calc_psnr(input, target, mean, std)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-3,
                     showwind=[8, 8]):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        summary(self.model, input_size=(self.image_channel, self.image_depth, self.image_height, self.image_width))
        print(self.model)
        showpixelvalue = 1.
        # 1、initialize loss function and optimizer
        self.model.apply(initialize_weights)
        lossFunc = self._loss_function(self.loss_name)
        opt = optim.AdamW(self.model.parameters(), lr=lr)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask, True)
        # 3、initialize a dictionary to store training history
        H = {"train_loss": [], "train_accuracy": [], "valdation_loss": [], "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.model.train()
            # 4.2、initialize the total training and validation loss
            totalTrainLoss = []
            totalTrainAccu = []
            totalValidationLoss = []
            totalValiadtionAccu = []
            trainshow = True
            # 4.3、loop over the training set
            for batch in train_loader:
                # x should tensor with shape (N,C,D,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,D,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                mean = batch['mean']
                std = batch['std']
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # perform a forward pass and calculate the training loss and accu
                pred_logit, _ = self.model(x)
                loss = lossFunc(pred_logit, y)
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                accu = self._accuracy_function(self.accuracyname, pred_logit, y, mean, std)
                # save_images
                if trainshow:
                    savepath = model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                    x0 = x[0] * std[0] + mean[0]
                    images = x0.detach().cpu().squeeze().numpy()
                    sitk_image = sitk.GetImageFromArray(images)
                    sitk.WriteImage(sitk_image, model_dir + "/trainsrc.nii.gz")
                    x0 = 255. * (x0 - torch.min(x0)) / (torch.max(x0) - torch.min(x0))
                    pd0 = pred_logit[0] * std[0] + mean[0]
                    pdimages = pd0.detach().cpu().squeeze().numpy()
                    sitk_image = sitk.GetImageFromArray(pdimages)
                    sitk.WriteImage(sitk_image, model_dir + "/trainpd.nii.gz")
                    pd0 = 255. * (pd0 - torch.min(pd0)) / (torch.max(pd0) - torch.min(pd0))
                    y0 = y[0] * std[0] + mean[0]
                    yimages = y0.detach().cpu().squeeze().numpy()
                    sitk_image = sitk.GetImageFromArray(yimages)
                    sitk.WriteImage(sitk_image, model_dir + "/trainmask.nii.gz")
                    y0 = 255. * (y0 - torch.min(y0)) / (torch.max(y0) - torch.min(y0))
                    save_images3dregression(x0, pd0, y0, showwind, savepath, pixelvalue=showpixelvalue)
                    trainshow = False
                # add the loss to the total training loss so far
                totalTrainLoss.append(loss.cpu().detach().numpy())
                totalTrainAccu.append(accu.cpu().detach().numpy())
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.model.eval()
            trainshow = True
            with torch.no_grad():
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    mean = batch['mean']
                    std = batch['std']
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred_logit, _ = self.model(x)
                    loss = lossFunc(pred_logit, y)
                    accu = self._accuracy_function(self.accuracyname, pred_logit, y, mean, std)
                    if trainshow:
                        # save_images
                        savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                        x0 = x[0] * std[0] + mean[0]
                        images = x0.detach().cpu().squeeze().numpy()
                        sitk_image = sitk.GetImageFromArray(images)
                        sitk.WriteImage(sitk_image, model_dir + "/valsrc.nii.gz")
                        x0 = 255. * (x0 - torch.min(x0)) / (torch.max(x0) - torch.min(x0))
                        pd0 = pred_logit[0] * std[0] + mean[0]
                        pdimages = pd0.detach().cpu().squeeze().numpy()
                        sitk_image = sitk.GetImageFromArray(pdimages)
                        sitk.WriteImage(sitk_image, model_dir + "/valpd.nii.gz")
                        pd0 = 255. * (pd0 - torch.min(pd0)) / (torch.max(pd0) - torch.min(pd0))
                        y0 = y[0] * std[0] + mean[0]
                        yimages = y0.detach().cpu().squeeze().numpy()
                        sitk_image = sitk.GetImageFromArray(yimages)
                        sitk.WriteImage(sitk_image, model_dir + "/valmask.nii.gz")
                        y0 = 255. * (y0 - torch.min(y0)) / (torch.max(y0) - torch.min(y0))
                        save_images3dregression(x0, pd0, y0, showwind, savepath, pixelvalue=showpixelvalue)
                        trainshow = False
                    totalValidationLoss.append(loss.cpu().detach().numpy())
                    totalValiadtionAccu.append(accu.cpu().detach().numpy())
            # 4.5、calculate the average training and validation loss
            avgTrainLoss = np.mean(np.stack(totalTrainLoss))
            avgValidationLoss = np.mean(np.stack(totalValidationLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # lr_scheduler.step(avgValidationLoss)
            # 4.6、update our training history
            H["train_loss"].append(avgTrainLoss)
            H["valdation_loss"].append(avgValidationLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("Train loss: {:.5f}, Train accu: {:.5f}，validation loss: {:.5f}, validation accu: {:.5f}".format(
                avgTrainLoss, avgTrainAccu, avgValidationLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/Loss', avgTrainLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/loss', avgValidationLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            # serialize best model to disk
            MODEL_PATH = os.path.join(model_dir, "unet_%d.pth" % e)
            torch.save(self.model.state_dict(), MODEL_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_loss"], H["valdation_loss"], "train_loss", "valdation_loss", "loss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.model.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            output, _ = self.model(img)
            probs = output[0]
            full_mask_np = probs.detach().cpu().squeeze().numpy()
            return full_mask_np

    def inference(self, imagesitk, newSize=(96, 96, 96)):
        # resize image and normalization,should rewrite
        _, resizeimagesitk = resize_image_itkwithsize(imagesitk, newSize, imagesitk.GetSize(), sitk.sitkLinear)
        imageresize = sitk.GetArrayFromImage(resizeimagesitk)
        mean = imageresize.mean()
        std = imageresize.std()
        imageresize = (imageresize - mean) / std
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        out_mask = self.predict(imageresize)
        out_mask = out_mask * std + mean
        # resize mask to src image size,should rewrite
        out_mask_sitk = sitk.GetImageFromArray(out_mask)
        out_mask_sitk.SetOrigin(resizeimagesitk.GetOrigin())
        out_mask_sitk.SetSpacing(resizeimagesitk.GetSpacing())
        out_mask_sitk.SetDirection(resizeimagesitk.GetDirection())
        _, final_out_mask_sitk = resize_image_itkwithsize(out_mask_sitk, imagesitk.GetSize(), newSize,
                                                          sitk.sitkLinear)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def inference_patch(self, imagesitk, newSpacing=(0.5, 0.5, 0.5)):
        # resize image and normalization,should rewrite
        # _, resizeimagesitk = resize_image_itk(imagesitk, newSpacing, imagesitk.GetSpacing(), sitk.sitkLinear)
        imageresize = sitk.GetArrayFromImage(imagesitk)
        # transpose (D,H,W,C) order to (C,D,H,W) order
        D, H, W = np.shape(imageresize)[0], np.shape(imageresize)[1], np.shape(imageresize)[2]
        imageresize = np.reshape(imageresize, (D, H, W, 1))
        imageresize = np.transpose(imageresize, (3, 0, 1, 2))
        # predict patch
        stepx = self.image_width // 2
        stepy = self.image_height // 2
        stepz = self.image_depth // 2
        out_mask = np.zeros((D, H, W))
        out_mask_weight = np.zeros((D, H, W))
        for z in range(0, D, stepz):
            for y in range(0, H, stepy):
                for x in range(0, W, stepx):
                    x_min = x
                    x_max = x_min + self.image_width
                    if x_max > W:
                        x_max = W
                        x_min = W - self.image_width
                    y_min = y
                    y_max = y_min + self.image_height
                    if y_max > H:
                        y_max = H
                        y_min = H - self.image_height
                    z_min = z
                    z_max = z_min + self.image_depth
                    if z_max > D:
                        z_max = D
                        z_min = D - self.image_depth
                    patch_xs = imageresize[:, z_min:z_max, y_min:y_max, x_min:x_max]
                    mean = patch_xs.mean()
                    std = patch_xs.std()
                    patch_xs = (patch_xs - mean) / std
                    predictresult = self.predict(patch_xs)
                    predictresult = predictresult * std + mean
                    out_mask[z_min:z_max, y_min:y_max, x_min:x_max] = out_mask[
                                                                      z_min:z_max,
                                                                      y_min:y_max,
                                                                      x_min:x_max] + predictresult.copy()
                    out_mask_weight[z_min:z_max, y_min:y_max, x_min:x_max] = out_mask_weight[
                                                                             z_min:z_max,
                                                                             y_min:y_max,
                                                                             x_min:x_max] + 1.
        # resize mask to src image size,should rewrite
        out_mask = out_mask / out_mask_weight
        out_mask = np.around(out_mask)
        out_mask[out_mask < 0] = 0
        final_out_mask_sitk = sitk.GetImageFromArray(out_mask)
        final_out_mask_sitk.SetOrigin(imagesitk.GetOrigin())
        final_out_mask_sitk.SetSpacing(imagesitk.GetSpacing())
        final_out_mask_sitk.SetDirection(imagesitk.GetDirection())
        return final_out_mask_sitk

    def clear_GPU_cache(self):
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
