import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
from network import Network_3D_Unet
from data_process import train_preprocess_lessMemoryMulStacks, trainset
from utils import save_yaml

#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=40, help="number of training epochs")
parser.add_argument('--GPU', type=str, default='0,1', help="the index of GPU you will use for computation")

parser.add_argument('--batch_size', type=int, default=2, help="batch size")
parser.add_argument('--img_w', type=int, default=150, help="the width of image patch")
parser.add_argument('--img_h', type=int, default=150, help="the height of image patch")
parser.add_argument('--img_s', type=int, default=150, help="the length of image patch")

parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")
parser.add_argument('--normalize_factor', type=int, default=1, help='normalize factor')
parser.add_argument('--fmap', type=int, default=16, help='number of feature maps')

parser.add_argument('--output_dir', type=str, default='./results', help="output directory")
parser.add_argument('--datasets_folder', type=str, default='train', help="A folder containing files for training")
parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")
parser.add_argument('--pth_path', type=str, default='pth', help="pth file root path")
parser.add_argument('--select_img_num', type=int, default=100000, help='select the number of images used for training')
parser.add_argument('--train_datasets_size', type=int, default=4000, help='datasets size for training')
opt = parser.parse_args()

# default image gap is 0.5*image_dim
# opt.gap_s (image gap) is the distance between two adjacent patches
opt.gap_s=int(opt.img_s*0.5)
opt.gap_w=int(opt.img_w*0.5)
opt.gap_h=int(opt.img_h*0.5)
opt.ngpu=str(opt.GPU).count(',')+1
print('\033[1;31mTraining parameters -----> \033[0m')
print(opt)

########################################################################################################################
if not os.path.exists(opt.output_dir): 
    os.mkdir(opt.output_dir)
current_time = opt.datasets_folder+'_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
output_path = opt.output_dir + '/' + current_time
pth_path = 'pth//'+ current_time
if not os.path.exists(pth_path): 
    os.mkdir(pth_path)

yaml_name = pth_path+'//para.yaml'
save_yaml(opt, yaml_name)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
batch_size = opt.batch_size
lr = opt.lr

name_list, noise_img_all, coordinate_list, stack_index = train_preprocess_lessMemoryMulStacks(opt)
# print('name_list -----> ',name_list)
########################################################################################################################
L1_pixelwise = torch.nn.L1Loss()
L2_pixelwise = torch.nn.MSELoss()

denoise_generator = Network_3D_Unet(in_channels = 1,
                                    out_channels = 1,
                                    f_maps=opt.fmap,
                                    final_sigmoid = True)

if torch.cuda.is_available():
    denoise_generator = denoise_generator.cuda()
    denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(opt.ngpu))
    print('\033[1;31mUsing {} GPU for training -----> \033[0m'.format(torch.cuda.device_count()))
    L2_pixelwise.cuda()
    L1_pixelwise.cuda()
########################################################################################################################
optimizer_G = torch.optim.Adam(denoise_generator.parameters(),
                                lr=opt.lr, betas=(opt.b1, opt.b2))

########################################################################################################################
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
prev_time = time.time()

########################################################################################################################
time_start=time.time()

# start training
for epoch in range(0, opt.n_epochs):
    train_data = trainset(name_list, coordinate_list, noise_img_all,stack_index)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    for iteration, (input, target) in enumerate(trainloader):
        input=input.cuda()
        target = target.cuda()
        real_A=input
        real_B=target
        real_A = Variable(real_A)
        #print('real_A shape -----> ', real_A.shape)
        #print('real_B shape -----> ',real_B.shape)
        fake_B = denoise_generator(real_A)
        L1_loss = L1_pixelwise(fake_B, real_B)
        L2_loss = L2_pixelwise(fake_B, real_B)
        ################################################################################################################
        optimizer_G.zero_grad()
        # Total loss
        Total_loss =  0.5*L1_loss + 0.5*L2_loss
        Total_loss.backward()
        optimizer_G.step()
        ################################################################################################################
        batches_done = epoch * len(trainloader) + iteration
        batches_left = opt.n_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
        prev_time = time.time()

        if iteration%1 == 0:
            time_end=time.time()
            print('\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f, L1 Loss: %.2f, L2 Loss: %.2f] [ETA: %s] [Time cost: %.2d s]        ' 
            % (
                epoch+1,
                opt.n_epochs,
                iteration+1,
                len(trainloader),
                Total_loss.item(),
                L1_loss.item(),
                L2_loss.item(),
                time_left,
                time_end-time_start
            ), end=' ')

        if (iteration+1)%len(trainloader) == 0:
            print('\n', end=' ')

        ################################################################################################################
        # save model
        if (iteration + 1) % (len(trainloader)) == 0:
            model_save_name = pth_path + '//E_' + str(epoch+1).zfill(2) + '_Iter_' + str(iteration+1).zfill(4) + '.pth'
            if isinstance(denoise_generator, nn.DataParallel): 
                torch.save(denoise_generator.module.state_dict(), model_save_name)  # parallel
            else:
                torch.save(denoise_generator.state_dict(), model_save_name)         # not parallel
