import numpy as np
import os
import tifffile as tiff
import random
import math
import torch
from torch.utils.data import Dataset

class trainset(Dataset):
    def __init__(self,name_list,coordinate_list,noise_img_all,stack_index):
        self.name_list = name_list
        self.coordinate_list=coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index

    def __getitem__(self, index):
        #fn = self.images[index]
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
        
        input=torch.from_numpy(np.expand_dims(input, 0))
        target=torch.from_numpy(np.expand_dims(target, 0))
        return input, target

    def __len__(self):
        return len(self.name_list)


class testset(Dataset):
    def __init__(self,name_list,coordinate_list,noise_img):
        self.name_list = name_list
        self.coordinate_list=coordinate_list
        self.noise_img = noise_img

    def __getitem__(self, index):
        #fn = self.images[index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        noise_patch = self.noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch=torch.from_numpy(np.expand_dims(noise_patch, 0))
        #target = self.target[index]
        return noise_patch,single_coordinate

    def __len__(self):
        return len(self.name_list)


def singlebatch_test_save(single_coordinate,output_image,raw_image):
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

    aaaa = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    bbbb = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    return aaaa,bbbb,stack_start_w,stack_end_w,stack_start_h,stack_end_h,stack_start_s,stack_end_s


def multibatch_test_save(single_coordinate,id,output_image,raw_image):
    stack_start_w_id = single_coordinate['stack_start_w'].numpy()
    stack_start_w = int(stack_start_w_id[id])
    stack_end_w_id = single_coordinate['stack_end_w'].numpy()
    stack_end_w=int(stack_end_w_id[id])
    patch_start_w_id = single_coordinate['patch_start_w'].numpy()
    patch_start_w=int(patch_start_w_id[id])
    patch_end_w_id = single_coordinate['patch_end_w'].numpy()
    patch_end_w=int(patch_end_w_id[id])

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

    output_image_id=output_image[id]
    raw_image_id=raw_image[id]
    aaaa = output_image_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
    bbbb = raw_image_id[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

    return aaaa,bbbb,stack_start_w,stack_end_w,stack_start_h,stack_end_h,stack_start_s,stack_end_s


def train_preprocess(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s*2
    gap_h = img_h*0.75
    gap_w = img_w*0.75
    gap_s2 = img_s2*0.75
    im_folder = 'datasets//'+args.datasets_folder

    name_list = []
    train_raw = []
    train_GT = []
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,int((whole_h-img_h+gap_h)/gap_h)):
            for y in range(0,int((whole_w-img_w+gap_w)/gap_w)):
                for z in range(0,int((whole_s-img_s2+gap_s2)/gap_s2)):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                    noise_patch1 = noise_im[init_s:end_s:2,init_h:end_h,init_w:end_w]
                    noise_patch2 = noise_im[init_s+1:end_s:2,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    train_raw.append(noise_patch1.transpose(1,2,0))
                    train_GT.append(noise_patch2.transpose(1,2,0))
                    name_list.append(patch_name)
    return train_raw, train_GT, name_list, noise_im

def test_preprocess(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s
    im_folder = 'datasets//'+args.datasets_folder

    name_list = []
    train_raw = []
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,int((whole_h-img_h+gap_h)/gap_h)):
            for y in range(0,int((whole_w-img_w+gap_w)/gap_w)):
                for z in range(0,int((whole_s-img_s2+gap_s2)/gap_s2)):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                    noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
    return train_raw, name_list, noise_im

def test_preprocess_lessMemory (args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s
    im_folder = 'datasets//'+args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list={}
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        if noise_im.shape[0]>args.test_datasize:
            noise_im = noise_im[0:args.test_datasize,:,:]
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,int((whole_h-img_h+gap_h)/gap_h)):
            for y in range(0,int((whole_w-img_w+gap_w)/gap_w)):
                for z in range(0,int((whole_s-img_s2+gap_s2)/gap_s2)):
                    single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    coordinate_list[patch_name] = single_coordinate
    return  name_list, noise_im, coordinate_list

def get_gap_s(args, img, stack_num):
    whole_w = img.shape[2]
    whole_h = img.shape[1]
    whole_s = img.shape[0]
    # print('whole_w -----> ',whole_w)
    # print('whole_h -----> ',whole_h)
    # print('whole_s -----> ',whole_s)
    w_num = math.floor((whole_w-args.img_w)/args.gap_w)+1
    h_num = math.floor((whole_h-args.img_h)/args.gap_h)+1
    s_num = math.ceil(args.train_datasets_size/w_num/h_num/stack_num)
    # print('w_num -----> ',w_num)
    # print('h_num -----> ',h_num)
    # print('s_num -----> ',s_num)
    gap_s = math.floor((whole_s-args.img_s*2)/(s_num-1))
    # print('gap_s -----> ',gap_s)
    return gap_s

def train_preprocess_lessMemory(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s*2
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s*2
    im_folder = args.datasets_path+'//'+args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list={}
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0]>args.select_img_num:
            noise_im = noise_im[0:args.select_img_num,:,:]
        gap_s2 = get_gap_s(args, noise_im)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,int((whole_h-img_h+gap_h)/gap_h)):
            for y in range(0,int((whole_w-img_w+gap_w)/gap_w)):
                for z in range(0,int((whole_s-img_s2+gap_s2)/gap_s2)):
                    single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    coordinate_list[patch_name] = single_coordinate
    return  name_list, noise_im, coordinate_list

def train_preprocess_lessMemoryMulStacks(args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s*2
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s*2
    im_folder = args.datasets_path + '//' + args.datasets_folder

    name_list = []
    coordinate_list={}
    stack_index = []
    noise_im_all = []
    ind = 0;
    print('\033[1;31mImage list for training -----> \033[0m')
    stack_num = len(list(os.walk(im_folder, topdown=False))[-1][-1])
    print('Total number -----> ', stack_num)
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        print(im_name)
        im_dir = im_folder+ '//' + im_name
        noise_im = tiff.imread(im_dir)
        if noise_im.shape[0]>args.select_img_num:
            noise_im = noise_im[0:args.select_img_num,:,:]
        gap_s2 = get_gap_s(args, noise_im, stack_num)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor 
        noise_im_all.append(noise_im)
        
        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,int((whole_h-img_h+gap_h)/gap_h)):
            for y in range(0,int((whole_w-img_w+gap_w)/gap_w)):
                for z in range(0,int((whole_s-img_s2+gap_s2)/gap_s2)):
                    single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_'+im_name.replace('.tif','')+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    coordinate_list[patch_name] = single_coordinate
                    stack_index.append(ind)
        ind = ind + 1;
    return  name_list, noise_im_all, coordinate_list, stack_index

def test_preprocess_lessMemoryPadding (args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s
    im_folder = 'datasets//'+args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list={}
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        raw_noise_im = tiff.imread(im_dir)
        if raw_noise_im.shape[0]>args.test_datasize:
            raw_noise_im = raw_noise_im[0:args.test_datasize,:,:]
        raw_noise_im = (raw_noise_im-raw_noise_im.min()).astype(np.float32)/args.normalize_factor

        print('raw_noise_im shape -----> ',raw_noise_im.shape)
        noise_im_w = math.ceil((raw_noise_im.shape[2]-img_w)/gap_w)*gap_w+img_w
        noise_im_h = math.ceil((raw_noise_im.shape[1]-img_h)/gap_h)*gap_h+img_h
        noise_im_s = math.ceil((raw_noise_im.shape[0]-img_s2)/gap_s2)*gap_s2+img_s2
        noise_im = np.zeros([noise_im_s,noise_im_h,noise_im_w])
        noise_im[0:raw_noise_im.shape[0], 0:raw_noise_im.shape[1], 0:raw_noise_im.shape[2]]=raw_noise_im
        noise_im = noise_im.astype(np.float32)
        print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,int((whole_h-img_h+gap_h)/gap_h)):
            for y in range(0,int((whole_w-img_w+gap_w)/gap_w)):
                for z in range(0,int((whole_s-img_s2+gap_s2)/gap_s2)):
                    single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s
                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    coordinate_list[patch_name] = single_coordinate
    return  name_list, noise_im, coordinate_list, raw_noise_im

def test_preprocess_lessMemoryNoTail (args):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s
    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s2 - gap_s2)/2
    im_folder = args.datasets_path+'//'+args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list={}
    for im_name in list(os.walk(im_folder, topdown=False))[-1][-1]:
        # print('im_name -----> ',im_name)
        im_dir = im_folder+'//'+im_name
        noise_im = tiff.imread(im_dir)
        # print('noise_im shape -----> ',noise_im.shape)
        # print('noise_im max -----> ',noise_im.max())
        # print('noise_im min -----> ',noise_im.min())
        if noise_im.shape[0]>args.test_datasize:
            noise_im = noise_im[0:args.test_datasize,:,:]
        noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

        whole_w = noise_im.shape[2]
        whole_h = noise_im.shape[1]
        whole_s = noise_im.shape[0]

        num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
        num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
        num_s = math.ceil((whole_s-img_s2+gap_s2)/gap_s2)
        # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
        # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
        # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
        for x in range(0,num_h):
            for y in range(0,num_w):
                for z in range(0,num_s):
                    single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                    if x != (num_h-1):
                        init_h = gap_h*x
                        end_h = gap_h*x + img_h
                    elif x == (num_h-1):
                        init_h = whole_h - img_h
                        end_h = whole_h

                    if y != (num_w-1):
                        init_w = gap_w*y
                        end_w = gap_w*y + img_w
                    elif y == (num_w-1):
                        init_w = whole_w - img_w
                        end_w = whole_w

                    if z != (num_s-1):
                        init_s = gap_s2*z
                        end_s = gap_s2*z + img_s2
                    elif z == (num_s-1):
                        init_s = whole_s - img_s2
                        end_s = whole_s
                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s

                    if y == 0:
                        single_coordinate['stack_start_w'] = y*gap_w
                        single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                        single_coordinate['patch_start_w'] = 0
                        single_coordinate['patch_end_w'] = img_w-cut_w
                    elif y == num_w-1:
                        single_coordinate['stack_start_w'] = whole_w-img_w+cut_w
                        single_coordinate['stack_end_w'] = whole_w
                        single_coordinate['patch_start_w'] = cut_w
                        single_coordinate['patch_end_w'] = img_w
                    else:
                        single_coordinate['stack_start_w'] = y*gap_w+cut_w
                        single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                        single_coordinate['patch_start_w'] = cut_w
                        single_coordinate['patch_end_w'] = img_w-cut_w

                    if x == 0:
                        single_coordinate['stack_start_h'] = x*gap_h
                        single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                        single_coordinate['patch_start_h'] = 0
                        single_coordinate['patch_end_h'] = img_h-cut_h
                    elif x == num_h-1:
                        single_coordinate['stack_start_h'] = whole_h-img_h+cut_h
                        single_coordinate['stack_end_h'] = whole_h
                        single_coordinate['patch_start_h'] = cut_h
                        single_coordinate['patch_end_h'] = img_h
                    else:
                        single_coordinate['stack_start_h'] = x*gap_h+cut_h
                        single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                        single_coordinate['patch_start_h'] = cut_h
                        single_coordinate['patch_end_h'] = img_h-cut_h

                    if z == 0:
                        single_coordinate['stack_start_s'] = z*gap_s2
                        single_coordinate['stack_end_s'] = z*gap_s2+img_s2-cut_s
                        single_coordinate['patch_start_s'] = 0
                        single_coordinate['patch_end_s'] = img_s2-cut_s
                    elif z == num_s-1:
                        single_coordinate['stack_start_s'] = whole_s-img_s2+cut_s
                        single_coordinate['stack_end_s'] = whole_s
                        single_coordinate['patch_start_s'] = cut_s
                        single_coordinate['patch_end_s'] = img_s2
                    else:
                        single_coordinate['stack_start_s'] = z*gap_s2+cut_s
                        single_coordinate['stack_end_s'] = z*gap_s2+img_s2-cut_s
                        single_coordinate['patch_start_s'] = cut_s
                        single_coordinate['patch_end_s'] = img_s2-cut_s

                    # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                    patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                    # train_raw.append(noise_patch1.transpose(1,2,0))
                    name_list.append(patch_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    coordinate_list[patch_name] = single_coordinate
    return  name_list, noise_im, coordinate_list


    # stack_start_w ,stack_end_w ,patch_start_w ,patch_end_w ,
    # stack_start_h ,stack_end_h ,patch_start_h ,patch_end_h ,
    # stack_start_s ,stack_end_s ,patch_start_s ,patch_end_s

def test_preprocess_lessMemoryNoTail_chooseOne (args, N):
    img_h = args.img_h
    img_w = args.img_w
    img_s2 = args.img_s
    gap_h = args.gap_h
    gap_w = args.gap_w
    gap_s2 = args.gap_s
    cut_w = (img_w - gap_w)/2
    cut_h = (img_h - gap_h)/2
    cut_s = (img_s2 - gap_s2)/2
    im_folder = args.datasets_path+'//'+args.datasets_folder

    name_list = []
    # train_raw = []
    coordinate_list={}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    # print(img_list)

    im_name = img_list[N]

    im_dir = im_folder+'//'+im_name
    noise_im = tiff.imread(im_dir)
    # print('noise_im shape -----> ',noise_im.shape)
    # print('noise_im max -----> ',noise_im.max())
    # print('noise_im min -----> ',noise_im.min())
    if noise_im.shape[0]>args.test_datasize:
        noise_im = noise_im[0:args.test_datasize,:,:]
    noise_im = (noise_im-noise_im.min()).astype(np.float32)/args.normalize_factor

    whole_w = noise_im.shape[2]
    whole_h = noise_im.shape[1]
    whole_s = noise_im.shape[0]

    num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
    num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
    num_s = math.ceil((whole_s-img_s2+gap_s2)/gap_s2)
    # print('int((whole_h-img_h+gap_h)/gap_h) -----> ',int((whole_h-img_h+gap_h)/gap_h))
    # print('int((whole_w-img_w+gap_w)/gap_w) -----> ',int((whole_w-img_w+gap_w)/gap_w))
    # print('int((whole_s-img_s2+gap_s2)/gap_s2) -----> ',int((whole_s-img_s2+gap_s2)/gap_s2))
    for x in range(0,num_h):
        for y in range(0,num_w):
            for z in range(0,num_s):
                single_coordinate={'init_h':0, 'end_h':0, 'init_w':0, 'end_w':0, 'init_s':0, 'end_s':0}
                if x != (num_h-1):
                    init_h = gap_h*x
                    end_h = gap_h*x + img_h
                elif x == (num_h-1):
                    init_h = whole_h - img_h
                    end_h = whole_h

                if y != (num_w-1):
                    init_w = gap_w*y
                    end_w = gap_w*y + img_w
                elif y == (num_w-1):
                    init_w = whole_w - img_w
                    end_w = whole_w

                if z != (num_s-1):
                    init_s = gap_s2*z
                    end_s = gap_s2*z + img_s2
                elif z == (num_s-1):
                    init_s = whole_s - img_s2
                    end_s = whole_s
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y*gap_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = img_w-cut_w
                elif y == num_w-1:
                    single_coordinate['stack_start_w'] = whole_w-img_w+cut_w
                    single_coordinate['stack_end_w'] = whole_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w
                else:
                    single_coordinate['stack_start_w'] = y*gap_w+cut_w
                    single_coordinate['stack_end_w'] = y*gap_w+img_w-cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = img_w-cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x*gap_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = img_h-cut_h
                elif x == num_h-1:
                    single_coordinate['stack_start_h'] = whole_h-img_h+cut_h
                    single_coordinate['stack_end_h'] = whole_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h
                else:
                    single_coordinate['stack_start_h'] = x*gap_h+cut_h
                    single_coordinate['stack_end_h'] = x*gap_h+img_h-cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = img_h-cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z*gap_s2
                    single_coordinate['stack_end_s'] = z*gap_s2+img_s2-cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = img_s2-cut_s
                elif z == num_s-1:
                    single_coordinate['stack_start_s'] = whole_s-img_s2+cut_s
                    single_coordinate['stack_end_s'] = whole_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s2
                else:
                    single_coordinate['stack_start_s'] = z*gap_s2+cut_s
                    single_coordinate['stack_end_s'] = z*gap_s2+img_s2-cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = img_s2-cut_s

                # noise_patch1 = noise_im[init_s:end_s,init_h:end_h,init_w:end_w]
                patch_name = args.datasets_folder+'_x'+str(x)+'_y'+str(y)+'_z'+str(z)
                # train_raw.append(noise_patch1.transpose(1,2,0))
                name_list.append(patch_name)
                # print(' single_coordinate -----> ',single_coordinate)
                coordinate_list[patch_name] = single_coordinate

    return  name_list, noise_im, coordinate_list
