import numpy as np
import os
import tifffile as tiff
import random
import math
import torch
from torch.utils.data import Dataset
from skimage import io



def random_transform(input, target):
    """
    The function for data augmentation. Randomly select one method among five
    transformation methods (including rotation and flip) or do not use data
    augmentation.

    Args:
        input, target : the input and target patch before data augmentation
    Return:
        input, target : the input and target patch after data augmentation
    """
    p_trans = random.randrange(6)
    if p_trans == 0:  # no transformation
        input = input
        target = target
    elif p_trans == 1:  # left rotate 90
        input = np.rot90(input, k=1, axes=(1, 2))
        target = np.rot90(target, k=1, axes=(1, 2))
    elif p_trans == 2:  # left rotate 180
        input = np.rot90(input, k=2, axes=(1, 2))
        target = np.rot90(target, k=2, axes=(1, 2))
    elif p_trans == 3:  # left rotate 270
        input = np.rot90(input, k=3, axes=(1, 2))
        target = np.rot90(target, k=3, axes=(1, 2))
    elif p_trans == 4:  # horizontal flip
        input = input[:, :, ::-1]
        target = target[:, :, ::-1]
    elif p_trans == 5:  # vertical flip
        input = input[:, ::-1, :]
        target = target[:, ::-1, :]
    return input, target


class trainset(Dataset):
    """
    Train set generator for pytorch training

    """

    def __init__(self, name_list, coordinate_list, noise_img_all, stack_index):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index

    def __getitem__(self, index):
        """
        For temporal stacks with a small lateral size or short recording period, sub-stacks can be
        randomly cropped from the original stack to augment the training set according to the record
        coordinate. Then, interlaced frames of each sub-stack are extracted to form two 3D tiles.
        One of them serves as the input and the other serves as the target for network training
        Args:
            index : the index of 3D patchs used for training
        Return:
            input, target : the consecutive frames of the 3D noisy patch serve as the input and target of the network
        """
        stack_index = self.stack_index[index]
        noise_img = self.noise_img_all[stack_index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        input = noise_img[init_s:end_s:2, init_h:end_h, init_w:end_w]
        target = noise_img[init_s + 1:end_s:2, init_h:end_h, init_w:end_w]
        p_exc = random.random()  # generate a random number determinate whether swap input and target
        if p_exc < 0.5:
            input, target = random_transform(input, target)
        else:
            temp = input
            input = target
            target = temp  # Swap input and target
            input, target = random_transform(input, target)

        input = torch.from_numpy(np.expand_dims(input, 0).copy())
        target = torch.from_numpy(np.expand_dims(target, 0).copy())
        return input, target

    def __len__(self):
        return len(self.name_list)


class testset(Dataset):
    """
    Test set generator for pytorch inference

    """

    def __init__(self, name_list, coordinate_list, noise_img):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img = noise_img

    def __getitem__(self, index):
        """
        Generate the sub-stacks of the noisy image.
        Args:
            index : the index of 3D patch used for testing
        Return:
            noise_patch : the sub-stacks of the noisy image
            single_coordinate : the specific coordinate of sub-stacks in the noisy image for stitching all sub-stacks
        """
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        noise_patch = self.noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch = torch.from_numpy(np.expand_dims(noise_patch, 0))
        return noise_patch, single_coordinate

    def __len__(self):
        return len(self.name_list)


def get_gap_t(args, img, stack_num):
    whole_x = img.shape[2]
    whole_y = img.shape[1]
    whole_t = img.shape[0]
    print('whole_x -----> ', whole_x)
    print('whole_y -----> ', whole_y)
    print('whole_t -----> ', whole_t)
    w_num = math.floor((whole_x - args.patch_x) / args.gap_x) + 1
    h_num = math.floor((whole_y - args.patch_y) / args.gap_y) + 1
    s_num = math.ceil(args.train_datasets_size / w_num / h_num / stack_num)
    # print('w_num -----> ',w_num)
    # print('h_num -----> ',h_num)
    # print('s_num -----> ',s_num)
    gap_t = math.floor((whole_t - args.patch_t * 2) / (s_num - 1))
    # print('gap_t -----> ',gap_t)
    return gap_t


def train_preprocess_lessMemoryMulStacks(args):
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t * 2
    gap_y = args.gap_y
    gap_x = args.gap_x
    im_folder = args.datasets_path + '//' + args.datasets_folder

    name_list = []
    coordinate_list = {}
    stack_index = []
    noise_im_all = []
    ind = 0;
    print('\033[1;31mImage list for training -----> \033[0m')
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    print('Total stack number -----> ', stack_num)

    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        print(im_name)
        im_dir = im_folder + '//' + im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0] > args.select_img_num:
            noise_im = noise_im[0:args.select_img_num, :, :]
        gap_t2 = get_gap_t(args, noise_im, stack_num)
        args.gap_t = gap_t2
        # print('gap_t2 -----> ',gap_t2)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = noise_im.astype(np.float32) / args.scale_factor  # no preprocessing
        # noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.scale_factor 
        noise_im_all.append(noise_im)

        whole_x = noise_im.shape[2]
        whole_y = noise_im.shape[1]
        whole_t = noise_im.shape[0]
        # print('int((whole_y-patch_y+gap_y)/gap_y) -----> ',int((whole_y-patch_y+gap_y)/gap_y))
        # print('int((whole_x-patch_x+gap_x)/gap_x) -----> ',int((whole_x-patch_x+gap_x)/gap_x))
        # print('int((whole_t-patch_t2+gap_t2)/gap_t2) -----> ',int((whole_t-patch_t2+gap_t2)/gap_t2))
        for x in range(0, int((whole_y - patch_y + gap_y) / gap_y)):
            for y in range(0, int((whole_x - patch_x + gap_x) / gap_x)):
                for z in range(0, int((whole_t - patch_t2 + gap_t2) / gap_t2)):
                    single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                    init_s = gap_t2 * z
                    end_s = gap_t2 * z + patch_t2
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder + '_' + im_name.replace('.tif', '') + '_x' + str(x) + '_y' + str(
                        y) + '_z' + str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    coordinate_list[patch_name] = single_coordinate
                    stack_index.append(ind)
        ind = ind + 1;
    return name_list, noise_im_all, coordinate_list, stack_index


def singlebatch_test_save(single_coordinate, output_image, raw_image):
    """
    Subtract overlapping regions (both the lateral and temporal overlaps) from the output sub-stacks (if the batch size equal to 1).

    Args:
        single_coordinate : the coordinate dict of the image
        output_image : the output sub-stack of the network
        raw_image : the noisy sub-stack
    Returns:
        output_patch : the output patch after subtract the overlapping regions
        raw_patch :  the raw patch after subtract the overlapping regions
        stack_start_ : the start coordinate of the patch in whole stack
        stack_end_ : the end coordinate of the patch in whole stack
    """
    stack_start_w = int(single_coordinate['stack_start_w'])
    stack_end_w = int(single_coordinate['stack_end_w'])
    patch_start_w = int(single_coordinate['patch_start_w'])
    patch_end_w = int(single_coordinate['patch_end_w'])

    stack_start_h = int(single_coordinate['stack_start_h'])
    stack_end_h = int(single_coordinate['stack_end_h'])
    patch_start_h = int(single_coordinate['patch_start_h'])
    patch_end_h = int(single_coordinate['patch_end_h'])

    stack_start_s = int(single_coordinate['stack_start_s'])
    stack_end_s = int(single_coordinate['stack_end_s'])
    patch_start_s = int(single_coordinate['patch_start_s'])
    patch_end_s = int(single_coordinate['patch_end_s'])

    output_patch = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    raw_patch = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    return output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def multibatch_test_save(single_coordinate, id, output_image, raw_image):
    """
    Subtract overlapping regions (both the lateral and temporal overlaps) from the output sub-stacks. (if the batch size larger than 1).

    Args:
        single_coordinate : the coordinate dict of the image
        output_image : the output sub-stack of the network
        raw_image : the noisy sub-stack
    Returns:
        output_patch : the output patch after subtract the overlapping regions
        raw_patch :  the raw patch after subtract the overlapping regions
        stack_start_ : the start coordinate of the patch in whole stack
        stack_end_ : the end coordinate of the patch in whole stack
    """
    stack_start_w_id = single_coordinate['stack_start_w'].numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = single_coordinate['stack_end_w'].numpy()
    stack_end_w = int(stack_end_w_id[id])
    patch_start_w_id = single_coordinate['patch_start_w'].numpy()
    patch_start_w = int(patch_start_w_id[id])
    patch_end_w_id = single_coordinate['patch_end_w'].numpy()
    patch_end_w = int(patch_end_w_id[id])

    stack_start_h_id = single_coordinate['stack_start_h'].numpy()
    stack_start_h = int(stack_start_h_id[id])
    stack_end_h_id = single_coordinate['stack_end_h'].numpy()
    stack_end_h = int(stack_end_h_id[id])
    patch_start_h_id = single_coordinate['patch_start_h'].numpy()
    patch_start_h = int(patch_start_h_id[id])
    patch_end_h_id = single_coordinate['patch_end_h'].numpy()
    patch_end_h = int(patch_end_h_id[id])

    stack_start_s_id = single_coordinate['stack_start_s'].numpy()
    stack_start_s = int(stack_start_s_id[id])
    stack_end_s_id = single_coordinate['stack_end_s'].numpy()
    stack_end_s = int(stack_end_s_id[id])
    patch_start_s_id = single_coordinate['patch_start_s'].numpy()
    patch_start_s = int(patch_start_s_id[id])
    patch_end_s_id = single_coordinate['patch_end_s'].numpy()
    patch_end_s = int(patch_end_s_id[id])

    output_image_id = output_image[id]
    raw_image_id = raw_image[id]
    output_patch = output_image_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    raw_patch = raw_image_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

    return output_patch, raw_patch, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s


def test_preprocess_lessMemoryNoTail_chooseOne(args, N):
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t
    gap_y = args.gap_y
    gap_x = args.gap_x
    gap_t2 = args.gap_t
    cut_w = (patch_x - gap_x) / 2
    cut_h = (patch_y - gap_y) / 2
    cut_s = (patch_t2 - gap_t2) / 2
    im_folder = args.datasets_path + '//' + args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list = {}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    # print(img_list)

    im_name = img_list[N]

    im_dir = im_folder + '//' + im_name
    noise_im = tiff.imread(im_dir)
    # print('noise_im shape -----> ',noise_im.shape)
    # print('noise_im max -----> ',noise_im.max())
    # print('noise_im min -----> ',noise_im.min())
    if noise_im.shape[0] > args.test_datasize:
        noise_im = noise_im[0:args.test_datasize, :, :]
    noise_im = noise_im.astype(np.float32) / args.scale_factor
    # noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.scale_factor

    whole_x = noise_im.shape[2]
    whole_y = noise_im.shape[1]
    whole_t = noise_im.shape[0]

    num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
    num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
    num_s = math.ceil((whole_t - patch_t2 + gap_t2) / gap_t2)
    # print('int((whole_y-patch_y+gap_y)/gap_y) -----> ',int((whole_y-patch_y+gap_y)/gap_y))
    # print('int((whole_x-patch_x+gap_x)/gap_x) -----> ',int((whole_x-patch_x+gap_x)/gap_x))
    # print('int((whole_t-patch_t2+gap_t2)/gap_t2) -----> ',int((whole_t-patch_t2+gap_t2)/gap_t2))
    for x in range(0, num_h):
        for y in range(0, num_w):
            for z in range(0, num_s):
                single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                if x != (num_h - 1):
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                elif x == (num_h - 1):
                    init_h = whole_y - patch_y
                    end_h = whole_y

                if y != (num_w - 1):
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                elif y == (num_w - 1):
                    init_w = whole_x - patch_x
                    end_w = whole_x

                if z != (num_s - 1):
                    init_s = gap_t2 * z
                    end_s = gap_t2 * z + patch_t2
                elif z == (num_s - 1):
                    init_s = whole_t - patch_t2
                    end_s = whole_t
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y * gap_x
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x - cut_w
                elif y == num_w - 1:
                    single_coordinate['stack_start_w'] = whole_x - patch_x + cut_w
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x
                else:
                    single_coordinate['stack_start_w'] = y * gap_x + cut_w
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x - cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x * gap_y
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = patch_y - cut_h
                elif x == num_h - 1:
                    single_coordinate['stack_start_h'] = whole_y - patch_y + cut_h
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y
                else:
                    single_coordinate['stack_start_h'] = x * gap_y + cut_h
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y - cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z * gap_t2
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s
                elif z == num_s - 1:
                    single_coordinate['stack_start_s'] = whole_t - patch_t2 + cut_s
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2
                else:
                    single_coordinate['stack_start_s'] = z * gap_t2 + cut_s
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s

                # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                patch_name = args.datasets_folder + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                name_list.append(patch_name)
                # print(' single_coordinate -----> ',single_coordinate)
                coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im, coordinate_list


def test_preprocess_chooseOne(args, img_id):
    """
    Choose one original noisy stack and partition it into thousands of 3D sub-stacks (patch) with the setting
    overlap factor in each dimension.

    Args:
        args : the train object containing input params for partition
        img_id : the id of the test image
    Returns:
        name_list : the coordinates of 3D patch are indexed by the patch name in name_list
        noise_im : the original noisy stacks
        coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack
        im_name : the file name of the noisy stacks

    """

    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t2 = args.patch_t
    gap_y = args.gap_y
    gap_x = args.gap_x
    gap_t2 = int(args.patch_t * (1 - args.overlap_factor))
    cut_w = (patch_x - gap_x) / 2
    cut_h = (patch_y - gap_y) / 2
    cut_s = (patch_t2 - gap_t2) / 2
    im_folder = args.datasets_path

    name_list = []
    coordinate_list = {}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()

    im_name = img_list[img_id]


    im_dir = im_folder + '//' + im_name
    noise_im = tiff.imread(im_dir)
    img_mean = noise_im.mean()
    # print('noise_im max -----> ',noise_im.max())
    # print('noise_im min -----> ',noise_im.min())
    if noise_im.shape[0] > args.test_datasize:
        noise_im = noise_im[0:args.test_datasize, :, :]
    if args.print_img_name:
       print('Testing image name -----> ', im_name)
       print('Testing image shape -----> ', noise_im.shape)
    # Minus mean before training
    noise_im = noise_im.astype(np.float32)/args.scale_factor
    noise_im = noise_im-img_mean
    # No preprocessing
    # noise_im = noise_im.astype(np.float32) / args.scale_factor
    # noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.scale_factor

    whole_x = noise_im.shape[2]
    whole_y = noise_im.shape[1]
    whole_t = noise_im.shape[0]

    num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
    num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
    num_s = math.ceil((whole_t - patch_t2 + gap_t2) / gap_t2)
    # print('int((whole_y-patch_y+gap_y)/gap_y) -----> ',int((whole_y-patch_y+gap_y)/gap_y))
    # print('int((whole_x-patch_x+gap_x)/gap_x) -----> ',int((whole_x-patch_x+gap_x)/gap_x))
    # print('int((whole_t-patch_t2+gap_t2)/gap_t2) -----> ',int((whole_t-patch_t2+gap_t2)/gap_t2))
    for x in range(0, num_h):
        for y in range(0, num_w):
            for z in range(0, num_s):
                single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                if x != (num_h - 1):
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                elif x == (num_h - 1):
                    init_h = whole_y - patch_y
                    end_h = whole_y

                if y != (num_w - 1):
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                elif y == (num_w - 1):
                    init_w = whole_x - patch_x
                    end_w = whole_x

                if z != (num_s - 1):
                    init_s = gap_t2 * z
                    end_s = gap_t2 * z + patch_t2
                elif z == (num_s - 1):
                    init_s = whole_t - patch_t2
                    end_s = whole_t
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y * gap_x
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x - cut_w
                elif y == num_w - 1:
                    single_coordinate['stack_start_w'] = whole_x - patch_x + cut_w
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x
                else:
                    single_coordinate['stack_start_w'] = y * gap_x + cut_w
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x - cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x * gap_y
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = patch_y - cut_h
                elif x == num_h - 1:
                    single_coordinate['stack_start_h'] = whole_y - patch_y + cut_h
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y
                else:
                    single_coordinate['stack_start_h'] = x * gap_y + cut_h
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y - cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z * gap_t2
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s
                elif z == num_s - 1:
                    single_coordinate['stack_start_s'] = whole_t - patch_t2 + cut_s
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2
                else:
                    single_coordinate['stack_start_s'] = z * gap_t2 + cut_s
                    single_coordinate['stack_end_s'] = z * gap_t2 + patch_t2 - cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t2 - cut_s

                # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                patch_name = args.datasets_name + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                name_list.append(patch_name)
                # print(' single_coordinate -----> ',single_coordinate)
                coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im, coordinate_list, im_name, img_mean
