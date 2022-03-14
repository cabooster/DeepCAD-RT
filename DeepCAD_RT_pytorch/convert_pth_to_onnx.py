import os
import torch
import torch.nn as nn
import argparse
import time
from deepcad.network import Network_3D_Unet
#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--GPU', type=str, default='0', help="the index of GPU you will use for computation")
parser.add_argument('--pth_path', type=str, default='pth', help="pth file root path")
parser.add_argument('--denoise_model', type=str, default='calcium-mouse-neuron-full_202201111604_200_40', help='A folder containing models to be tested')
parser.add_argument('--patch_x', type=int, default=200, help="the width of 3D patches (patch size in x)")
parser.add_argument('--patch_y', type=int, default=200, help="the width of 3D patches (patch size in y)")
parser.add_argument('--patch_t', type=int, default=40, help="the width of 3D patches (patch size in t)")

opt = parser.parse_args()
opt.ngpu=str(opt.GPU).count(',')+1
print('\033[1;31mParameters -----> \033[0m')
print(opt)

########################################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
model_path = opt.pth_path + '//' + opt.denoise_model
model_list = list(os.walk(model_path, topdown=False))[-1][-1]
model_list.sort()

for i in range(len(model_list)):
    aaa = model_list[i]
    if '.yaml' in aaa:
        yaml_name = model_list[i]
        del model_list[i]

##############################################################################################################################################################
# network architecture and GPU access
denoise_generator = Network_3D_Unet(in_channels=1,
                                    out_channels=1,
                                    f_maps=16,
                                    final_sigmoid=True)
if torch.cuda.is_available():
    print('\033[1;31mUsing {} GPU for testing -----> \033[0m'.format(torch.cuda.device_count()))
    denoise_generator = denoise_generator.cuda()
    denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(opt.ngpu))
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
##############################################################################################################################################################
time_start = time.time()
# Start processing
for pth_index in range(len(model_list)):
    aaa = model_list[pth_index]
    if '.pth' in aaa:
        pth_name = model_list[pth_index]

        # load model
        model_name = opt.pth_path + '//' + opt.denoise_model + '//' + pth_name
        if isinstance(denoise_generator, nn.DataParallel):
            denoise_generator.module.load_state_dict(torch.load(model_name))  # parallel
            denoise_generator.eval()
        else:
            denoise_generator.load_state_dict(torch.load(model_name))  # not parallel
            denoise_generator.eval()

        model = denoise_generator.cuda()
        input_name = ['input']
        output_name = ['output']

        # input = torch.randn(1, 1, 80, 200, 200, requires_grad=True).cuda()

        # input = torch.randn(1, 1, 150, 150, 150, requires_grad=True).cuda()

        input = torch.randn(1, 1, opt.patch_t, opt.patch_x, opt.patch_y, requires_grad=True).cuda()
        torch.onnx.export(model.module, input, pth_name.replace('.pth', '.onnx'), export_params=True,input_names=input_name, output_names=output_name,opset_version=11, verbose=True)

time_end = time.time()
print('Using time--->',time_end - time_start,'s')

