"""
This file will demonstrate pipeline for training microscopy data using the DeepCAD-RT algorithm.
The demo shows how to construct the params and call the relevant functions for training DeepCAD-RT network.
The demo will automatically download tif file for demo training.
See inside for details.

* This demo is also available as a jupyter notebook (see demo_train_pipeline.ipynb) and Colab notebook (see
DeepCAD_RT_demo_colab.ipynb)

More information can be found in the companion paper.
"""

from deepcad.train_collection import training_class
from deepcad.movie_display import display
from deepcad.utils import get_first_filename,download_demo

# %% Select file(s) to be processed (download if not present)
download_demo_file = True
if download_demo_file:
    file_name='fish_localbrain' # select the demo file to be trained (e.g. 'ATP_3D', 'fish_localbrain', 'NP_3D', ...)
    datasets_path, _ = download_demo(download_filename=file_name)
else:
    datasets_path = 'datasets/fish_localbrain_demo'  # folder containing tif files for training

# %% First setup some parameters for training
n_epochs = 10               # the number of training epochs
GPU = '1'                   # the index of GPU used for computation (e.g. '0', '0,1', '0,1,2')
train_datasets_size = 6000  # dataset size for training (the number of patches)
patch_xy = 150              # the width and height of 3D patches
patch_t = 150               # the time dimension of 3D patches
overlap_factor = 0.25       # the overlap factor between two adjacent patches
pth_dir = './pth'           # pth file and visualization result file path
num_workers = 4             # if you use Windows system, set this to 0.

# %% Setup some parameters for result visualization during training period (optional)
visualize_images_per_epoch = True  # choose whether to show inference performance after each epoch
save_test_images_per_epoch = True  # choose whether to save inference image after each epoch in pth path

# %% Play the demo noise movie (optional)
# playing the first noise movie using opencv.
display_images = True

if display_images:
    display_filename = get_first_filename(datasets_path)
    print('\033[1;31mDisplaying the first raw file -----> \033[0m')
    print(display_filename)
    display_length = 300  # the frames number of the noise movie
    # normalize the image and display
    display(display_filename, display_length=display_length, norm_min_percent=1, norm_max_percent=98)

train_dict = {
    # dataset dependent parameters
    'patch_x': patch_xy,
    'patch_y': patch_xy,
    'patch_t': patch_t,
    'overlap_factor':overlap_factor,
    'scale_factor': 1,                  # the factor for image intensity scaling
    'select_img_num': 100000,           # select the number of images used for training (use all frames by default)
    'train_datasets_size': train_datasets_size,
    'datasets_path': datasets_path,
    'pth_dir': pth_dir,
    # network related parameters
    'n_epochs': n_epochs,
    'lr': 0.00005,                       # initial learning rate
    'b1': 0.5,                           # Adam: bata1
    'b2': 0.999,                         # Adam: bata2
    'fmap': 16,                          # the number of feature maps
    'GPU': GPU,
    'num_workers': num_workers,
    'visualize_images_per_epoch': visualize_images_per_epoch,
    'save_test_images_per_epoch': save_test_images_per_epoch
}
# %%% Training preparation
# first we create a training class object with the specified parameters
tc = training_class(train_dict)
# start the training process
tc.run()
