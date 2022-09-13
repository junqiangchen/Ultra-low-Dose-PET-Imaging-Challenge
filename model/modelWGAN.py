from .dataset import datasetModelRegressionwithopencv, datasetModelRegressionwithnpy
import torch.nn as nn
from .metric import calc_psnr
from torch.utils.data import DataLoader
import torch
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
import torch.autograd as autograd
from pathlib import Path
import os
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from networks import initialize_weights
import time
from tqdm import tqdm
from .visualization import plot_result, save_images2dregression, save_images3dregression
import cv2


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        # suppose x is your feature map with size N*C*H*W
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        # now x is of size N*C
        return x


class GeneratorUNet2d(nn.Module):
    """"
    GeneratorUNet2dnetwork
    """

    def __init__(self, in_channels, out_channels, init_features=16):
        super(GeneratorUNet2d, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = GeneratorUNet2d._block(self.in_channels, self.features, name="enc1", prob=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = GeneratorUNet2d._block(self.features, self.features * 2, name="enc2", prob=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = GeneratorUNet2d._block(self.features * 2, self.features * 4, name="enc3", prob=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = GeneratorUNet2d._block(self.features * 4, self.features * 8, name="enc4", prob=0.2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = GeneratorUNet2d._block(self.features * 8, self.features * 16, name="bottleneck", prob=0.2)
        self.upconv4 = nn.ConvTranspose2d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
        self.decoder4 = GeneratorUNet2d._block((self.features * 8) * 2, self.features * 8, name="dec4", prob=0.2)
        self.upconv3 = nn.ConvTranspose2d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
        self.decoder3 = GeneratorUNet2d._block((self.features * 4) * 2, self.features * 4, name="dec3", prob=0.2)
        self.upconv2 = nn.ConvTranspose2d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.decoder2 = GeneratorUNet2d._block((self.features * 2) * 2, self.features * 2, name="dec2", prob=0.2)
        self.upconv1 = nn.ConvTranspose2d(self.features * 2, self.features, kernel_size=2, stride=2)
        self.decoder1 = GeneratorUNet2d._block(self.features * 2, self.features, name="dec1", prob=0.2)
        self.conv = nn.Conv2d(in_channels=self.features, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out_logit = self.conv(dec1)

        return out_logit

    @staticmethod
    def _block(in_channels, features, name, prob=0.2):
        return nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=5,
                padding=2,
                stride=1,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(8, features)),
            (name + "drop1", nn.Dropout2d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
        ]))


class Discriminator2d(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=16):
        super(Discriminator2d, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = Discriminator2d._block(self.in_channels, self.features, name="enc1", prob=0.2)
        self.encoder2 = Discriminator2d._block(self.features, self.features * 2, name="enc2", prob=0.2)
        self.encoder3 = Discriminator2d._block(self.features * 2, self.features * 4, name="enc3", prob=0.2)
        self.encoder4 = Discriminator2d._block(self.features * 4, self.features * 8, name="enc4", prob=0.2)
        self.bottleneck = Discriminator2d._block(self.features * 8, self.features * 16, name="bottleneck", prob=0.2)

        self.avg = GlobalAveragePooling()

        self.fc_layers = nn.Sequential(
            nn.Linear(self.features * 16, 128),
            nn.Dropout(p=0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.out_channels),
        )

    def forward(self, x):
        x = self.bottleneck(self.encoder4(self.encoder3(self.encoder2(self.encoder1(x)))))
        x = self.avg(x)
        # print("x.shape", x.shape) # 1, 256
        x = self.fc_layers(x)
        return x

    @staticmethod
    def _block(in_channels, features, name, prob=0.2):
        return nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=5,
                padding=2,
                stride=2,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(8, features)),
            (name + "drop1", nn.Dropout2d(p=prob, inplace=True)),
            (name + "relu1", nn.LeakyReLU(0.2, inplace=True)),
        ]))


class WGAN2dModel(object):
    """
    WGAN2d with binary class,should rewrite the dataset class and inference fucntion
    """

    def __init__(self, image_height, image_width, image_channel, numclass, batch_size, inference=False,
                 model_path=None,
                 use_cuda=True):
        self.batch_size = batch_size
        self.accuracyname = 'PSNR'
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.numclass = numclass

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.Tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        self.generator = GeneratorUNet2d(self.image_channel, self.numclass)
        self.discriminator = Discriminator2d(self.image_channel, self.numclass)
        self.generator.to(device=self.device)
        self.discriminator.to(device=self.device)

        if inference:
            print(f'Loading model {model_path}')
            print(f'Using device {self.device}')
            self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded!')

    def _dataloder(self, images, labels, shuffle=False):
        """"
        train dataset shuffle is true,validation is false
        """
        dataset = datasetModelRegressionwithopencv(images, labels,
                                                   targetsize=(
                                                       self.image_channel, self.image_height,
                                                       self.image_width))
        # fow window num_workers is only zero,for linux num_workers can not zero
        # num_cpu = multiprocessing.cpu_count()
        num_cpu = 0
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=self.batch_size, num_workers=num_cpu,
                                pin_memory=True)
        return dataloader

    def _compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(self.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True,
                                  retain_graph=True, only_inputs=True, )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _loss_function(self, real_validity, fake_validity, gradient_penalty, lambda_gp=10):
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        g_loss = -torch.mean(fake_validity)
        return d_loss, g_loss

    def _accuracy_function(self, accuracyname, input, target, mean, std):
        if accuracyname is 'PSNR':
            return calc_psnr(input, target, mean, std)

    def trainprocess(self, trainimage, trainmask, validationimage, validationmask, model_dir, epochs=50, lr=1e-4):
        print("[INFO] training the network...")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        summary(self.generator, input_size=(self.image_channel, self.image_height, self.image_width))
        print(self.generator)
        summary(self.discriminator, input_size=(self.image_channel, self.image_height, self.image_width))
        print(self.discriminator)
        showpixelvalue = 1.
        if self.numclass > 1:
            showpixelvalue = showpixelvalue // (self.numclass - 1)
        # 1、initialize loss function and optimizer
        self.generator.apply(initialize_weights)
        self.discriminator.apply(initialize_weights)
        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, verbose=True)
        # 2、load data train and validation dataset
        train_loader = self._dataloder(trainimage, trainmask, True)
        val_loader = self._dataloder(validationimage, validationmask, True)
        # 3、initialize a dictionary to store training history
        H = {"train_Dloss": [], "train_Gloss": [], "train_accuracy": [], "valdation_Dloss": [], "valdation_Gloss": [],
             "valdation_accuracy": []}
        # 4、start loop training wiht epochs times
        startTime = time.time()
        # Tensorboard summary
        writer = SummaryWriter(log_dir=model_dir)
        for e in tqdm(range(epochs)):
            # 4.1、set the model in training mode
            self.generator.train()
            self.discriminator.train()
            # 4.2、initialize the total training and validation loss
            totalTrainDLoss = []
            totalTrainGLoss = []
            totalTrainAccu = []
            totalValidationDLoss = []
            totalValidationGLoss = []
            totalValiadtionAccu = []
            # 4.3、loop over the training set
            trainshow = True
            for i, batch in enumerate(train_loader):
                # x should tensor with shape (N,C,D,W,H)
                x = batch['image']
                # y should tensor with shape (N,C,D,W,H),
                # if have mutil label y should one-hot,if only one label,the C is one
                y = batch['label']
                y = torch.reshape(y, (-1, self.numclass, self.image_width, self.image_height))
                mean = batch['mean']
                std = batch['std']
                # send the input to the device
                x, y = x.to(self.device), y.to(self.device)
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # perform a forward pass and calculate the training loss and accu
                # Generate a batch of images
                optimizer_D.zero_grad()
                fake_imgs = self.generator(x)
                # Real images
                real_validity = self.discriminator(y)
                # Fake images
                fake_validity = self.discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = self._compute_gradient_penalty(self.discriminator, y.data, fake_imgs.data)
                d_loss, _ = self._loss_function(real_validity, fake_validity, gradient_penalty)
                # first, zero out any previously accumulated gradients,
                # then perform backpropagation,
                # and then update model parameters
                d_loss.backward()
                optimizer_D.step()
                optimizer_G.zero_grad()
                # Train the generator every n_critic steps
                if i % 5 == 0:
                    fake_imgs = self.generator(x)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_imgs)
                    _, g_loss = self._loss_function(real_validity, fake_validity, gradient_penalty)
                    g_loss.backward()
                    optimizer_G.step()
                accu = self._accuracy_function(self.accuracyname, fake_imgs, y, mean, std)
                if trainshow:
                    # save_images
                    savepath = model_dir + '/' + str(e + 1) + "_Train_EPOCH_"
                    save_images2dregression(x[0] * std[0] + mean[0],
                                            fake_imgs[0] * std[0] + mean[0],
                                            y[0] * std[0] + mean[0],
                                            savepath, pixelvalue=showpixelvalue)
                    trainshow = False
                # add the loss to the total training loss so far
                totalTrainDLoss.append(d_loss.cpu().detach().numpy())
                totalTrainGLoss.append(g_loss.cpu().detach().numpy())
                totalTrainAccu.append(accu.cpu().detach().numpy())
            # 4.4、switch off autograd and loop over the validation set
            # set the model in evaluation mode
            self.generator.eval()
            self.discriminator.eval()
            trainshow = True
            with torch.no_grad():
                # loop over the validation set
                for batch in val_loader:
                    # x should tensor with shape (N,C,W,H)
                    x = batch['image']
                    # y should tensor with shape (N,C,W,H)
                    y = batch['label']
                    y = torch.reshape(y, (-1, self.numclass, self.image_width, self.image_height))
                    mean = batch['mean']
                    std = batch['std']
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    fake_imgs = self.generator(x)
                    # Real images
                    real_validity = self.discriminator(y)
                    # Fake images
                    fake_validity = self.discriminator(fake_imgs)
                    # Gradient penalty
                    d_loss, g_loss = self._loss_function(real_validity, fake_validity, 0)
                    accu = self._accuracy_function(self.accuracyname, fake_imgs, y, mean, std)
                    if trainshow:
                        # save_images
                        savepath = model_dir + '/' + str(e + 1) + "_Val_EPOCH_"
                        save_images2dregression(x[0] * std[0] + mean[0],
                                                fake_imgs[0] * std[0] + mean[0],
                                                y[0] * std[0] + mean[0],
                                                savepath, pixelvalue=showpixelvalue)
                        trainshow = False
                    totalValidationDLoss.append(d_loss.cpu().detach().numpy())
                    totalValidationGLoss.append(g_loss.cpu().detach().numpy())
                    totalValiadtionAccu.append(accu.cpu().detach().numpy())
            # 4.5、calculate the average training and validation loss
            avgTrainDLoss = np.mean(np.stack(totalTrainDLoss))
            avgValidationDLoss = np.mean(np.stack(totalValidationDLoss))
            avgTrainGLoss = np.mean(np.stack(totalTrainGLoss))
            avgValidationGLoss = np.mean(np.stack(totalValidationGLoss))
            avgTrainAccu = np.mean(np.stack(totalTrainAccu))
            avgValidationAccu = np.mean(np.stack(totalValiadtionAccu))
            # lr_scheduler.step(avgValidationLoss)
            # 4.6、update our training history
            H["train_Dloss"].append(avgTrainDLoss)
            H["valdation_Dloss"].append(avgValidationDLoss)
            H["train_Gloss"].append(avgTrainGLoss)
            H["valdation_Gloss"].append(avgValidationGLoss)
            H["train_accuracy"].append(avgTrainAccu)
            H["valdation_accuracy"].append(avgValidationAccu)
            # 4.7、print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
            print("train_Dloss: {:.5f},train_Gloss: {:.5f}，train_accuracy: {:.5f}".format(avgTrainDLoss, avgTrainGLoss,
                                                                                          avgTrainAccu))
            print("valdation_Dloss: {:.5f}, valdation_Gloss: {:.5f}，validation accu: {:.5f}".format(
                avgValidationDLoss, avgValidationGLoss, avgValidationAccu))
            # Record training loss and accuracy for each phase
            writer.add_scalar('Train/DLoss', avgTrainDLoss, e + 1)
            writer.add_scalar('Train/GLoss', avgTrainGLoss, e + 1)
            writer.add_scalar('Train/accu', avgTrainAccu, e + 1)
            writer.add_scalar('Valid/Dloss', avgValidationDLoss, e + 1)
            writer.add_scalar('Valid/Gloss', avgValidationGLoss, e + 1)
            writer.add_scalar('Valid/accu', avgValidationAccu, e + 1)
            writer.flush()
            # 4.8、save best_validation_dsc model params
            generator_PATH = os.path.join(model_dir, "generator_%d.pth" % e)
            discriminator_PATH = os.path.join(model_dir, "discriminator_%d.pth" % e)
            torch.save(self.generator.state_dict(), generator_PATH)
            torch.save(self.discriminator.state_dict(), discriminator_PATH)
        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
        # 5、plot the training loss
        plot_result(model_dir, H["train_Dloss"], H["train_Gloss"], "train_Dloss", "train_Gloss", "trainloss")
        plot_result(model_dir, H["valdation_Dloss"], H["valdation_Gloss"], "valdation_Dloss", "valdation_Gloss",
                    "validationloss")
        plot_result(model_dir, H["train_accuracy"], H["valdation_accuracy"], "train_accuracy", "valdation_accuracy",
                    "accuracy")
        self.clear_GPU_cache()

    def predict(self, full_img):
        # 1、clear cache
        self.clear_GPU_cache()
        # 2、set model eval
        self.generator.eval()
        # 3、convet numpy image to tensor
        img = torch.as_tensor(full_img).float().contiguous()
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)
        # 4、predict result
        with torch.no_grad():
            output = self.generator(img)
            probs = output[0]
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
