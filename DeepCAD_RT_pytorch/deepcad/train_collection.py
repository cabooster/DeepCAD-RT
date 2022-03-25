import os
import datetime

import numpy as np
import yaml

from .network import Network_3D_Unet
import tifffile as tiff
import random
import math
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import datetime
from .data_process import trainset, test_preprocess_chooseOne, testset, multibatch_test_save, singlebatch_test_save
from skimage import io
from .movie_display import test_img_display,display_img


class training_class():
    """
    Class implementing training process
    """


    def __init__(self, params_dict):
        """
        Constructor class for training process

        Args:
           params_dict: dict
               The collection of training params set by users
        Returns:
           self

        """
        self.overlap_factor = 0.5
        self.datasets_path = ''
        self.n_epochs = 20
        self.fmap = 16
        self.output_dir = './results'
        self.pth_dir = './pth'
        self.onnx_dir = './onnx'
        self.batch_size = 1
        self.patch_t = 150
        self.patch_x = 150
        self.patch_y = 150
        self.gap_y = 115
        self.gap_x = 115
        self.gap_t = 115
        self.lr = 0.00001
        self.b1 = 0.5
        self.b2 = 0.999
        self.GPU = '0'
        self.ngpu = 1
        self.num_workers = 0
        self.scale_factor = 1
        self.train_datasets_size = 2000
        self.select_img_num = 1000
        self.test_datasize = 400  # how many slices to be tested (use the first image in the folder by default)
        self.visualize_images_per_epoch = False
        self.save_test_images_per_epoch = False
        self.colab_display = False
        self.result_display = ''
        self.set_params(params_dict)
    def run(self):
        """
        General function for training DeepCAD network.

        """
        # create some essential file for result storage
        self.prepare_file()
        # crop input tiff file into 3D patches
        self.train_preprocess_lessMemoryMulStacks()
        # save some essential training parameters in para.yaml
        self.save_yaml_train()
        # initialize denoise network with training parameters.
        self.initialize_network()
        # specifies the GPU for the training program.
        self.distribute_GPU()
        # start training and result visualization during training period (optional)
        self.train()


    def prepare_file(self):
        """
        Make data folder to store training results
        Important Fields:
            self.datasets_name: the sub folder of the dataset
            self.pth_path: the folder for pth file storage

        """
        if self.datasets_path[-1]!='/':
           self.datasets_name=self.datasets_path.split("/")[-1]
        else:
            self.datasets_name=self.datasets_path.split("/")[-2]
        # self.datasets_name = self.datasets_path.split("/")[-1]
        pth_name = self.datasets_name + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.pth_path = self.pth_dir + '/' + pth_name
        self.onnx_path = self.onnx_dir + '/' + pth_name
        if not os.path.exists(self.pth_path):
            os.makedirs(self.pth_path)
        if not os.path.exists(self.onnx_path):
            os.makedirs(self.onnx_path)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def set_params(self, params_dict):
        """
        Set the params set by user to the training class object and calculate some default parameters for training

        """
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))  # patch gap in x
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))  # patch gap in y
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))  # patch gap in t
        self.ngpu = str(self.GPU).count(',') + 1                    # check the number of GPU used for training
        self.batch_size = self.ngpu                                 # By default, the batch size is equal to the number of GPU for minimal memory consumption
        print('\033[1;31mTraining parameters -----> \033[0m')
        print(self.__dict__)


    def initialize_network(self):
        """
        Initialize U-Net 3D network, which is the main network architecture of DeepCAD

        Important Fields:
           self.fmap: the number of the feature map in U-Net 3D network.
           self.local_model: the denoise network

        """
        denoise_generator = Network_3D_Unet(in_channels=1,
                                            out_channels=1,
                                            f_maps=self.fmap,
                                            final_sigmoid=True)
        self.local_model = denoise_generator

    def get_gap_t(self):
        """
        Calculate the patch gap in t according to the size of input data and the patch gap in x and y

        Important Fields:
           self.gap_t: the patch gap in t.

        """
        w_num = math.floor((self.whole_x - self.patch_x) / self.gap_x) + 1
        h_num = math.floor((self.whole_y - self.patch_y) / self.gap_y) + 1
        s_num = math.ceil(self.train_datasets_size / w_num / h_num / self.stack_num)
        self.gap_t = math.floor((self.whole_t - self.patch_t * 2) / (s_num - 1))

    def train_preprocess_lessMemoryMulStacks(self):
        """
        The original noisy stack is partitioned into thousands of 3D sub-stacks (patch) with the setting
        overlap factor in each dimension.

        Important Fields:
           self.name_list : the coordinates of 3D patch are indexed by the patch name in name_list.
           self.coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack.
           self.stack_index : the index of the noisy stacks.
           self.noise_im_all : the collection of all noisy stacks.

        """
        self.name_list = []
        self.coordinate_list = {}
        self.stack_index = []
        self.noise_im_all = []
        ind = 0
        print('\033[1;31mImage list for training -----> \033[0m')
        self.stack_num = len(list(os.walk(self.datasets_path, topdown=False))[-1][-1])
        print('Total stack number -----> ', self.stack_num)

        for im_name in list(os.walk(self.datasets_path, topdown=False))[-1][-1]:
            print('Noise image name -----> ', im_name)
            im_dir = self.datasets_path + '//' + im_name
            noise_im = tiff.imread(im_dir)
            if noise_im.shape[0] > self.select_img_num:
                noise_im = noise_im[0:self.select_img_num, :, :]
            self.whole_x = noise_im.shape[2]
            self.whole_y = noise_im.shape[1]
            self.whole_t = noise_im.shape[0]
            print('Noise image shape -----> ', noise_im.shape)
            # Calculate real gap_t
            self.get_gap_t()
            # No preprocessing
            # noise_im = noise_im.astype(np.float32) / self.scale_factor
            # Minus mean before training
            noise_im = noise_im.astype(np.float32)/self.scale_factor
            noise_im = noise_im-noise_im.mean()

            self.noise_im_all.append(noise_im)
            patch_t2 = self.patch_t * 2
            for x in range(0, int((self.whole_y - self.patch_y + self.gap_y) / self.gap_y)):
                for y in range(0, int((self.whole_x - self.patch_x + self.gap_x) / self.gap_x)):
                    for z in range(0, int((self.whole_t - patch_t2 + self.gap_t) / self.gap_t)):
                        single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                        init_h = self.gap_y * x
                        end_h = self.gap_y * x + self.patch_y
                        init_w = self.gap_x * y
                        end_w = self.gap_x * y + self.patch_x
                        init_s = self.gap_t * z
                        end_s = self.gap_t * z + patch_t2
                        single_coordinate['init_h'] = init_h
                        single_coordinate['end_h'] = end_h
                        single_coordinate['init_w'] = init_w
                        single_coordinate['end_w'] = end_w
                        single_coordinate['init_s'] = init_s
                        single_coordinate['end_s'] = end_s
                        patch_name = self.datasets_name + '_' + im_name.replace('.tif', '') + '_x' + str(
                            x) + '_y' + str(y) + '_z' + str(z)
                        self.name_list.append(patch_name)
                        self.coordinate_list[patch_name] = single_coordinate
                        self.stack_index.append(ind)
            ind = ind + 1

    def save_yaml_train(self):
        """
        Save some essential params in para.yaml.

        """
        yaml_name = self.pth_path + '//para.yaml'
        para = {'n_epochs': 0, 'datasets_path': 0, 'overlap_factor': 0,
                'output_dir': 0, 'pth_path': 0, 'GPU': 0, 'batch_size': 0,
                'patch_x': 0, 'patch_y': 0, 'patch_t': 0, 'gap_y': 0, 'gap_x': 0,
                'gap_t': 0, 'lr': 0, 'b1': 0, 'b2': 0, 'fmap': 0, 'scale_factor': 0,
                'select_img_num': 0, 'train_datasets_size': 0}
        para["n_epochs"] = self.n_epochs
        para["datasets_path"] = self.datasets_path
        para["output_dir"] = self.output_dir
        para["pth_path"] = self.pth_path
        para["GPU"] = self.GPU
        para["batch_size"] = self.batch_size
        para["patch_x"] = self.patch_x
        para["patch_y"] = self.patch_y
        para["patch_t"] = self.patch_t
        para["gap_x"] = self.gap_x
        para["gap_y"] = self.gap_y
        para["gap_t"] = self.gap_t
        para["lr"] = self.lr
        para["b1"] = self.b1
        para["b2"] = self.b2
        para["fmap"] = self.fmap
        para["scale_factor"] = self.scale_factor
        para["select_img_num"] = self.select_img_num
        para["train_datasets_size"] = self.train_datasets_size
        para["overlap_factor"] = self.overlap_factor
        with open(yaml_name, 'w') as f:
            yaml.dump(para, f)

    def distribute_GPU(self):
        """
        Allocate the GPU for the training program. Print the using GPU information to the screen.
        For acceleration, multiple GPUs parallel training is recommended.

        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU)
        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()
            self.local_model = nn.DataParallel(self.local_model, device_ids=range(self.ngpu))
            print('\033[1;31mUsing {} GPU(s) for training -----> \033[0m'.format(torch.cuda.device_count()))

    def train(self):
        """
        Pytorch training workflow

        """
        optimizer_G = torch.optim.Adam(self.local_model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        prev_time = time.time()
        time_start = time.time()
        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()
        L2_pixelwise.cuda()
        L1_pixelwise.cuda()

        for epoch in range(0, self.n_epochs):
            train_data = trainset(self.name_list, self.coordinate_list, self.noise_im_all, self.stack_index)
            trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            for iteration, (input, target) in enumerate(trainloader):
                # The input volume and corresponding target volume from data loader to train the deep neural network
                input = input.cuda()
                target = target.cuda()
                real_A = input
                real_B = target
                real_A = Variable(real_A)
                fake_B = self.local_model(real_A)
                L1_loss = L1_pixelwise(fake_B, real_B)
                L2_loss = L2_pixelwise(fake_B, real_B)
                optimizer_G.zero_grad()

                # Calculate total loss
                Total_loss = 0.5 * L1_loss + 0.5 * L2_loss
                Total_loss.backward()
                optimizer_G.step()
                # Record and estimate the remaining time
                batches_done = epoch * len(trainloader) + iteration
                batches_left = self.n_epochs * len(trainloader) - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
                prev_time = time.time()

                if iteration % 1 == 0:
                    time_end = time.time()
                    print(
                        '\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f, L1 Loss: %.2f, L2 Loss: %.2f] [ETA: %s] [Time cost: %.2d s]     '
                        % (
                            epoch + 1,
                            self.n_epochs,
                            iteration + 1,
                            len(trainloader),
                            Total_loss.item(),
                            L1_loss.item(),
                            L2_loss.item(),
                            time_left,
                            time_end - time_start
                        ), end=' ')



                if (iteration + 1) % (len(trainloader)) == 0:
                    print('\n', end=' ')
                    # Save model at the end of every epoch
                    self.save_model(epoch, iteration)
                    # Start inference using the denoise model at the end of every epoch (optional)
                    if (self.visualize_images_per_epoch | self.save_test_images_per_epoch):
                        print('Testing model of epoch {} on the first noisy file ----->'.format(epoch + 1))
                        self.test(epoch, iteration)
                        print('\n', end=' ')
        print('Train finished. Save all models to disk.')
        if self.colab_display:
            result_img_list = []
            results_path = self.pth_path
            results_list = list(os.walk(results_path, topdown=False))[-1][-1]
            for i in range(len(results_list)):
              aaa = results_list[i]
              if '.tif' in aaa:
                 result_img_list.append(aaa)
            result_img_list.sort()
            self.result_display = results_path+'/'+result_img_list[-1]


    def save_model(self, epoch, iteration):
        """
        Model storage.
        Args:
           train_epoch : current train epoch number
           train_iteration : current train_iteration number
        """
        model_save_name = self.pth_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(iteration + 1).zfill(
            4) + '.pth'
        if isinstance(self.local_model, nn.DataParallel):
            torch.save(self.local_model.module.state_dict(), model_save_name)  # parallel
        else:
            torch.save(self.local_model.state_dict(), model_save_name)  # not parallel
        # covert pth to onnx
        onnx_save_name = self.onnx_path + '//E_' + str(epoch + 1).zfill(2) + '_Iter_' + str(iteration + 1).zfill(
            4) + '_Patch_'+ str(self.patch_x) + '_' + str(self.patch_y) + '_' + str(self.patch_t) + '.onnx'
        input_name = ['input']
        output_name = ['output']

        input = torch.randn(1, 1, self.patch_t, self.patch_x,  self.patch_y, requires_grad=True).cuda()
        torch.onnx.export(self.local_model.module, input, onnx_save_name, export_params=True,input_names=input_name, output_names=output_name,opset_version=11, verbose=False)


    def test(self, train_epoch, train_iteration):
        """
        Pytorch testing workflow
        Args:
            train_epoch : current train epoch number
            train_iteration : current train_iteration number
        """
        # Crop test file into 3D patches for inference
        self.print_img_name = True
        name_list, noise_img, coordinate_list, test_im_name, img_mean = test_preprocess_chooseOne(self, img_id=0)
        # Record the inference time
        prev_time = time.time()
        time_start = time.time()
        denoise_img = np.zeros(noise_img.shape)
        input_img = np.zeros(noise_img.shape)
        test_data = testset(name_list, coordinate_list, noise_img)
        testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
            # Pre-trained models are loaded into memory and the sub-stacks are directly fed into the model.
            noise_patch = noise_patch.cuda()
            real_A = noise_patch
            real_A = Variable(real_A)
            fake_B = self.local_model(real_A)

            # Determine approximate time left
            batches_done = iteration
            batches_left = 1 * len(testloader) - batches_done
            time_left_seconds = int(batches_left * (time.time() - prev_time))
            time_left = datetime.timedelta(seconds=time_left_seconds)
            prev_time = time.time()
            if iteration % 1 == 0:
                time_end = time.time()
                time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                print(
                    '\r [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                    % (
                        iteration + 1,
                        len(testloader),
                        time_cost,
                        time_left_seconds
                    ), end=' ')

            if (iteration + 1) % len(testloader) == 0:
                print('\n', end=' ')

            # Enhanced sub-stacks are sequentially output from the network
            output_image = np.squeeze(fake_B.cpu().detach().numpy())
            raw_image = np.squeeze(real_A.cpu().detach().numpy())
            if (output_image.ndim == 3):
                postprocess_turn = 1
            else:
                postprocess_turn = output_image.shape[0]

            # The final enhanced stack can be obtained by stitching all sub-stacks.
            if (postprocess_turn > 1):
                for id in range(postprocess_turn):
                    output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                        single_coordinate, id, output_image, raw_image)

                    output_patch=output_patch+img_mean
                    raw_patch=raw_patch+img_mean

                    denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                    input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = raw_patch
            else:
                output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                    single_coordinate, output_image, raw_image)

                output_patch=output_patch+img_mean
                raw_patch=raw_patch+img_mean

                denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = output_patch * (np.sum(raw_patch) / np.sum(output_patch)) ** 0.5
                input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = raw_patch

        # Stitching finish
        output_img = denoise_img.squeeze().astype(np.float32) * self.scale_factor
        del denoise_img

        # Normalize and display inference image
        if (self.visualize_images_per_epoch):
            print('Displaying the first denoised file ----->')
            display_length = self.test_datasize
            test_img_display(output_img, display_length=display_length, norm_min_percent=1, norm_max_percent=98)

        # Save inference image
        if (self.save_test_images_per_epoch):
            output_img = output_img[50:self.test_datasize-50, :, :]
            output_img = output_img.astype('int16')
            result_name = self.pth_path + '//' + test_im_name.replace('.tif', '') + '_' + 'E_' + str(
                train_epoch + 1).zfill(2) + '_Iter_' + str(train_iteration + 1).zfill(4) + '.tif'
            io.imsave(result_name, output_img, check_contrast=False)
