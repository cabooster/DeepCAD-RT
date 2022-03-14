import os

import matplotlib.pyplot as plt
import yaml
import gdown
import zipfile

plt.ioff()
plt.switch_backend('agg')


########################################################################################################################
def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


def save_yaml_train(opt, yaml_name):
    para = {'n_epochs': 0,
            'datasets_folder': 0,
            'datasets_path': 0,
            'output_dir': 0,
            'pth_path': 0,
            'GPU': 0,
            'batch_size': 0,
            'patch_x': 0,
            'patch_y': 0,
            'patch_t': 0,
            'gap_y': 0,
            'gap_x': 0,
            'gap_t': 0,
            'lr': 0,
            'b1': 0,
            'b2': 0,
            'fmap': 0,
            'scale_factor': 0,
            'select_img_num': 0,
            'train_datasets_size': 0}
    para["n_epochs"] = opt.n_epochs
    para["datasets_folder"] = opt.datasets_folder
    para["datasets_path"] = opt.datasets_path
    para["output_dir"] = opt.output_dir
    para["pth_path"] = opt.pth_path
    para["GPU"] = opt.GPU
    para["batch_size"] = opt.batch_size
    para["patch_x"] = opt.patch_x
    para["patch_y"] = opt.patch_y
    para["patch_t"] = opt.patch_t
    para["gap_x"] = opt.gap_x
    para["gap_y"] = opt.gap_y
    para["gap_t"] = opt.gap_t
    para["lr"] = opt.lr
    para["b1"] = opt.b1
    para["b2"] = opt.b2
    para["fmap"] = opt.fmap
    para["scale_factor"] = opt.scale_factor
    para["select_img_num"] = opt.select_img_num
    para["train_datasets_size"] = opt.train_datasets_size
    with open(yaml_name, 'w') as f:
        data = yaml.dump(para, f)


def save_yaml_test(opt, yaml_name):
    para = {'n_epochs': 0,
            'datasets_folder': 0,
            'datasets_path': 0,
            'output_dir': 0,
            'pth_path': 0,
            'GPU': 0,
            'batch_size': 0,
            'patch_x': 0,
            'patch_y': 0,
            'patch_t': 0,
            'gap_x': 0,
            'gap_y': 0,
            'gap_t': 0,
            'lr': 0,
            'b1': 0,
            'b2': 0,
            'fmap': 0,
            'scale_factor': 0,
            'denoise_model': 0,
            'test_datasize': 0}
    para["n_epochs"] = opt.n_epochs
    para["datasets_folder"] = opt.datasets_folder
    para["datasets_path"] = opt.datasets_path
    para["output_dir"] = opt.output_dir
    para["pth_path"] = opt.pth_path
    para["GPU"] = opt.GPU
    para["batch_size"] = opt.batch_size
    para["patch_x"] = opt.patch_x
    para["patch_y"] = opt.patch_y
    para["patch_t"] = opt.patch_t
    para["gap_x"] = opt.gap_x
    para["gap_y"] = opt.gap_y
    para["gap_t"] = opt.gap_t
    para["lr"] = opt.lr
    para["b1"] = opt.b1
    para["b2"] = opt.b2
    para["fmap"] = opt.fmap
    para["scale_factor"] = opt.scale_factor
    para["denoise_model"] = opt.denoise_model
    para["test_datasize"] = opt.test_datasize
    with open(yaml_name, 'w') as f:
        data = yaml.dump(para, f)


def read_yaml(opt, yaml_name):
    with open(yaml_name) as f:
        para = yaml.load(f, Loader=yaml.FullLoader)
        print(para)
        opt.n_epochspara = ["n_epochs"]
        # opt.datasets_folder = para["datasets_folder"]
        opt.output_dir = para["output_dir"]
        opt.batch_size = para["batch_size"]
        # opt.patch_t = para["patch_t"]
        # opt.patch_x = para["patch_x"]
        # opt.patch_y = para["patch_y"]
        # opt.gap_y = para["gap_y"]
        # opt.gap_x = para["gap_x"]
        # opt.gap_t = para["gap_t"]
        opt.lr = para["lr"]
        opt.fmap = para["fmap"]
        opt.b1 = para["b1"]
        para["b2"] = opt.b2
        para["scale_factor"] = opt.scale_factor


def name2index(opt, input_name, num_h, num_w, num_s):
    # print(input_name)
    name_list = input_name.split('_')
    # print(name_list)
    z_part = name_list[-1]
    # print(z_part)
    y_part = name_list[-2]
    # print(y_part)
    x_part = name_list[-3]
    # print(x_part)
    z_index = int(z_part.replace('z', ''))
    y_index = int(y_part.replace('y', ''))
    x_index = int(x_part.replace('x', ''))
    # print("x_index ---> ",x_index,"y_index ---> ", y_index,"z_index ---> ", z_index)

    cut_w = (opt.patch_x - opt.gap_x) / 2
    cut_h = (opt.patch_y - opt.gap_y) / 2
    cut_s = (opt.patch_t - opt.gap_t) / 2
    # print("z_index ---> ",cut_w, "cut_h ---> ",cut_h, "cut_s ---> ",cut_s)
    if x_index == 0:
        stack_start_w = x_index * opt.gap_x
        stack_end_w = x_index * opt.gap_x + opt.patch_x - cut_w
        patch_start_w = 0
        patch_end_w = opt.patch_x - cut_w
    elif x_index == num_w - 1:
        stack_start_w = x_index * opt.gap_x + cut_w
        stack_end_w = x_index * opt.gap_x + opt.patch_x
        patch_start_w = cut_w
        patch_end_w = opt.patch_x
    else:
        stack_start_w = x_index * opt.gap_x + cut_w
        stack_end_w = x_index * opt.gap_x + opt.patch_x - cut_w
        patch_start_w = cut_w
        patch_end_w = opt.patch_x - cut_w

    if y_index == 0:
        stack_start_h = y_index * opt.gap_y
        stack_end_h = y_index * opt.gap_y + opt.patch_y - cut_h
        patch_start_h = 0
        patch_end_h = opt.patch_y - cut_h
    elif y_index == num_h - 1:
        stack_start_h = y_index * opt.gap_y + cut_h
        stack_end_h = y_index * opt.gap_y + opt.patch_y
        patch_start_h = cut_h
        patch_end_h = opt.patch_y
    else:
        stack_start_h = y_index * opt.gap_y + cut_h
        stack_end_h = y_index * opt.gap_y + opt.patch_y - cut_h
        patch_start_h = cut_h
        patch_end_h = opt.patch_y - cut_h

    if z_index == 0:
        stack_start_s = z_index * opt.gap_t
        stack_end_s = z_index * opt.gap_t + opt.patch_t - cut_s
        patch_start_s = 0
        patch_end_s = opt.patch_t - cut_s
    elif z_index == num_s - 1:
        stack_start_s = z_index * opt.gap_t + cut_s
        stack_end_s = z_index * opt.gap_t + opt.patch_t
        patch_start_s = cut_s
        patch_end_s = opt.patch_t
    else:
        stack_start_s = z_index * opt.gap_t + cut_s
        stack_end_s = z_index * opt.gap_t + opt.patch_t - cut_s
        patch_start_s = cut_s
        patch_end_s = opt.patch_t - cut_s
    return int(stack_start_w), int(stack_end_w), int(patch_start_w), int(patch_end_w), \
           int(stack_start_h), int(stack_end_h), int(patch_start_h), int(patch_end_h), \
           int(stack_start_s), int(stack_end_s), int(patch_start_s), int(patch_end_s)


def get_first_filename(img_dir):
    """
    Acquire the the first filename of the image directory
    Args:
         img_dir : image directory name
    Return:
         first_filename : the first filename of the image directory
    """
    train_list = list(os.walk(img_dir, topdown=False))[-1][-1]
    first_filename = img_dir + '/' + train_list[0]
    return first_filename


def download_demo(download_filename):
    save_img_folder='datasets/'+download_filename+'_demo'
    save_pth_name = download_filename + '_best_model_demo'
    save_pth_folder='pth/'+save_pth_name

    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    if not os.path.exists(save_pth_folder):
        os.makedirs(save_pth_folder)
    file_dict = {'simulate_-2.51dBSNR_1000frames': [
        'https://zenodo.org/record/5790790/files/noise_2RPN_-2.51dBSNR_1000frames_.tif?download=1',
        'https://zenodo.org/record/5790790/files/E_13_Iter_6198.pth?download=1',
        'https://zenodo.org/record/5790790/files/para.yaml?download=1'],
        'ATP_2D': [
        'https://zenodo.org/record/5808571/files/zoom3.0_15Hz_0.65power_20umUnderrInjury_lowSNR_MC_corrBidir.tif?download=1',
        'https://zenodo.org/record/5808571/files/ATP_2D_E_20_Iter_6075.pth?download=1',
        'https://zenodo.org/record/5808571/files/ATP_2D_para.yaml?download=1'],
        'ATP_3D':['https://zenodo.org/record/5808571/files/zoom2.0_3DATP_30Hz_2umStep_60umRange_30Slice_0.1power_channel1_slice22.tif?download=1',
                  'https://zenodo.org/record/5808571/files/ATP_3D_E_20_Iter_6350.pth?download=1',
                  'https://zenodo.org/record/5808571/files/ATP_3D_para.yaml?download=1'],
        'NP_2D':['https://zenodo.org/record/5808571/files/02_neutrophil_0.25power_zoom4.0_10Hz_lowSNR.tif?download=1',
                 'https://zenodo.org/record/5808571/files/NP_2D_E_20_Iter_6150.pth?download=1',
                 'https://zenodo.org/record/5808571/files/NP_2D_para.yaml?download=1'],
        'NP_3D':['https://zenodo.org/record/5808571/files/zoom4.5_dz2um_slice15_power0.07_gan0.7_channel23_channel1_slice5.tif?download=1',
                 'https://zenodo.org/record/5808571/files/NP_3D_E_20_Iter_6075.pth?download=1',
                 'https://zenodo.org/record/5808571/files/NP_3D_para.yaml?download=1'],
        'mouse_spine':['https://zenodo.org/record/5808571/files/1_GCaMP6f_zoom4.5_50mWpower_40umdepth_30Hz_978x492x6500_lowSNR.tif?download=1',
                       'https://zenodo.org/record/5808571/files/mouse_spine_E_30_Iter_6060.pth?download=1',
                       'https://zenodo.org/record/5808571/files/mouse_spine_para.yaml?download=1'],
        'fish_localbrain': [
        'https://zenodo.org/record/5808571/files/fish_localbrain_demo.zip?download=1',
        'https://zenodo.org/record/5808571/files/fish_localbrain_E_20_Iter_6175.pth?download=1',
        'https://zenodo.org/record/5808571/files/fish_localbrain_para.yaml?download=1'],
        'fish_wholebrain': [
        'https://zenodo.org/record/5808571/files/fish_wholebrain_demo.zip?download=1',
        'https://zenodo.org/record/5808571/files/fish_wholebrain_E_20_Iter_6140.pth?download=1',
        'https://zenodo.org/record/5808571/files/fish_wholebrain_para.yaml?download=1'],
        'drosophila': [
        'https://zenodo.org/record/5808571/files/drosophila_demo.zip?download=1',
        'https://zenodo.org/record/5808571/files/drosophila_E_30_Iter_6016.pth?download=1',
        'https://zenodo.org/record/5808571/files/drosophila_para.yaml?download=1'],
                 }

    url_sum = file_dict[download_filename]

    if not os.path.exists(save_img_folder+ '/' + download_filename +'.tif'):
        if (download_filename == 'fish_localbrain') | (download_filename == 'fish_wholebrain') | (download_filename == 'drosophila'):
            zip_path='datasets/' + download_filename + '_demo.zip'
            gdown.download(url_sum[0], zip_path , quiet=False,resume=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('datasets')
            os.remove(zip_path)
        else:
            gdown.download(url_sum[0], save_img_folder+ '/' + download_filename +'.tif', quiet=False,resume=True)
    if not os.path.exists(save_pth_folder+ '/' + 'best_model.pth'):
        gdown.download(url_sum[1], save_pth_folder+ '/' + 'best_model.pth', quiet=False,resume=True)
    if not os.path.exists(save_pth_folder+ '/' + 'best_model.yaml'):
        gdown.download(url_sum[2], save_pth_folder+ '/' + 'best_model.yaml', quiet=False,resume=True)
    return save_img_folder,save_pth_name
