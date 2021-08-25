# DeepCAD-RT: Deep self-supervised learning for calcium imaging denoising

<img src="images/logo.PNG" width="800" align="middle">

:triangular_flag_on_post: This is the **upgraded version** of [the original DeepCAD Pytorch code](https://github.com/cabooster/DeepCAD). We have made extensive updates and the new features including:
- **More stable performance**
- **Much faster processing speed**
- **Much lower memory cost**
- **Improved pre- and post-processing**
- **Multi-GPU acceleration**
- **Less parameters**
- **...**

## Contents

- [Overview](#overview)
- [Directory structure](#directory-structure)
- [Pytorch code](#pytorch-code)
- [License](./LICENSE)
- [Citation](#citation)

## Overview

<img src="images/schematic.png" width="400" align="right">

Calcium imaging is inherently susceptible to detection noise especially when imaging with high frame rate or under low excitation dosage. However, calcium transients are highly dynamic, non-repetitive activities and a firing pattern cannot be captured twice. Clean images for supervised training of deep neural networks are not accessible. Here, we present DeepCAD, a **deep** self-supervised learning-based method for **ca**lcium imaging **d**enoising. Using our method, detection noise can be effectively removed and the accuracy of neuron extraction and spike inference can be highly improved.

DeepCAD is based on the insight that a deep learning network for image denoising can achieve satisfactory convergence even the target image used for training is another corrupted sampling of the same scene [[paper link]](https://arxiv.org/abs/1803.04189). We explored the temporal redundancy of calcium imaging and found that any two consecutive frames can be regarded as two independent samplings of the same underlying firing pattern. A single low-SNR stack is sufficient to be a complete training set for DeepCAD. Furthermore, to boost its performance on 3D temporal stacks, the input and output data are designed to be 3D volumes rather than 2D frames to fully incorporate the abundant information along time axis.

For more details, please see the companion paper where the method first appeared: 
["*Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising*"](https://www.nature.com/articles/s41592-021-01225-0).

## Directory structure

```
DeepCAD
|---DeepCAD_pytorch #Pytorch implementation of DeepCAD#
|---|---train.py
|---|---test.py
|---|---script.py
|---|---network.py
|---|---model_3DUnet.py
|---|---data_process.py
|---|---buildingblocks.py
|---|---utils.py
|---|---datasets
|---|---|---DataForPytorch # project_name #
|---|---|---|---data.tif
|---|---pth
|---|---|---ModelForPytorch
|---|---|---|---model.pth
|---|---results
|---|---|--- # test results#
```

## Pytorch code

### Our environment 

* Ubuntu 16.04 
* Python 3.6
* Pytorch 1.8.0
* NVIDIA GPU (GeForce RTX 3090) + CUDA (11.1)

### Environment configuration

* Create a virtual environment, install Pytorch and other dependencies. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from https://pytorch.org/get-started/previous-versions/.

```
$ conda create -n deepcad python=3.6
$ source activate deepcad
$ conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
$ conda install -c conda-forge matplotlib pyyaml tifffile scikit-image
```

### Download the source code

```
$ git clone git://github.com/cabooster/DeepCAD-RT
$ cd DeepCAD-RT
$ rm datasets/DataForPytorch/DownloadedData & rm pth/ModelForPytorch/DownloadedModel
```

### Training

Download the demo data(.tif file, 5.9 GB) [[DataForPytorch](https://drive.google.com/drive/folders/1w9v1SrEkmvZal5LH79HloHhz6VXSPfI_)] and put it into *./datasets/DataForPytorch/*

Run the **script.py** to start training.

```
$ source activate deepcad
$ python script.py train
```

The network snapshot after each training epoch will be automatically saved in *./pth/*. Parameters can be modified  as required in **script.py**.

```
os.system('python train.py --datasets_folder DataForPytorch --n_epochs 40 --GPU 0 --batch_size 1 --img_h 150 --img_w 150 --img_s 150 --train_datasets_size 3500')  

@parameters
--datasets_folder: the folder containing your training data (one or more stacks)
--n_epochs: the number of training epochs
--GPU: specify GPU(s) used for training. The format should be something like '0', '0,1', '1,2,3'...
--batch_size: batch size (should be equal to the number of GPUs) 
--img_h, --img_w, --img_s: patch size in three dimensions
--training_datasets_size: the number of training patches extracted from the input stack(s).
```

### Test

Run the **script.py** to start the test process. A pre-trained model has been uploaded to *./pth/ModelForPytorch*. Parameters saved in the .yaml file will be automatically loaded. Test results will be be automatically saved in *./results/*
```
$ source activate deepcad
$ python script.py test
```

All models in the `--denoise_model` folder will be tested and manual inspection should be made for **model screening**. Parameters can be modified  as required in **script.py**. 

```
os.system('python test.py --denoise_model ModelForPytorch --datasets_folder DataForPytorch --GPU 0 --batch_size 1 --test_datasize 300')

@parameters
--denoise_model: the folder containing all the pre-trained models.
--datasets_folder: the folder containing the testing data (one or more stacks).
--test_datasize: the number of frames used for testing. If this number is larger than input frame numbers, all frames will be tested
```

## Citation

If you use this code please cite the companion paper where the original method appeared: 

Li, X., Zhang, G., Wu, J. et al. Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising. Nat Methods (2021). [https://doi.org/10.1038/s41592-021-01225-0](https://www.nature.com/articles/s41592-021-01225-0)
