---
layout: page
title: DeepCAD-RT tutorial
---
<img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/logo.PNG?raw=true" width="700" align="right" />



## Content

- [1. Python source code](#1-python-source-code)
- [2. Jupyter notebook](#2-jupyter-notebook)
- [3. Colab notebook](#3-colab-notebook)
- [4. Matlab implementation for real-time processing](#4-matlab-implementation-for-real-time-processing)

## 1. Python source code

### 1.1 Our environment 

* Ubuntu 16.04 
* Python 3.6
* Pytorch 1.8.0
* NVIDIA GPU (GeForce RTX 3090) + CUDA (11.1)

### 1.2 Environment configuration

1. Create a virtual environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). 

   ```
   $ conda create -n deepcadrt python=3.6
   $ conda activate deepcadrt
   $ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. We made a pip installable realease of DeepCAD-RT [[pypi](https://pypi.org/project/deepcad/)]. You can install it by entering the following command:

   ```
   $ pip install deepcad
   ```

### 1.3 Download the source code

```
$ git clone git://github.com/cabooster/DeepCAD-RT
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/
```

### 1.4 Demos

To try out the Python code, please activate `deepcadrt` conda environment first:

```
$ conda activate deepcadrt
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/
```

**Example training**

To train a DeepCAD-RT model, we recommend starting with the demo script `demo_train_pipeline.py`. One demo dataset will be downloaded to the `DeepCAD_RT_pytorch/datasets` folder automatically. You can also download other data from [the companion webpage](https://cabooster.github.io/DeepCAD-RT/Datasets/) or use your own data by changing the training parameter `datasets_path`. 


```
python demo_train_pipeline.py
```

**Example testing**

To test the denoising performance with pre-trained models, you can run the demo script `demo_test_pipeline.py` . A demo dataset and its denoising model will be automatically downloaded to `DeepCAD_RT_pytorch/datasets` and `DeepCAD_RT_pytorch/pth`, respectively. You can change the dataset and the model by changing the parameters `datasets_path` and `denoise_model`.

```
python demo_test_pipeline.py
```

## 2. Jupyter notebook

We provide simple and user-friendly Jupyter notebooks to implement DeepCAD-RT. They are in the `DeepCAD_RT_pytorch/notebooks` folder. Before you launch the notebooks, please configure an environment following the instruction in [Environment configuration](#environment-configuration) . And then, you can launch the notebooks through the following commands:

```
$ conda activate deepcadrt
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/notebooks
$ jupyter notebook
```

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/deepcad8.png?raw=true" width="700" align="middle"></center> 

## 3. Colab notebook

We also provide a cloud-based notebook implemented with Google Colab. You can run DeepCAD-RT directly in your browser using a cloud GPU without configuring the environment. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cabooster/DeepCAD-RT/blob/main/DeepCAD_RT_pytorch/notebooks/DeepCAD_RT_demo_colab.ipynb)

*Note: Colab notebook needs longer time to train and test because of the limited GPU performance offered by Colab.*

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/deepcad7.png?raw=true" width="700" align="middle"></center> 

## 4. Matlab implementation for real-time processing

To achieve real-time denoising, DeepCAD-RT was optimally deployed on GPU using TensorRT (Nvidia) for further acceleration and memory reduction. We also designed a sophisticated time schedule for multi-thread processing. Based on a two-photon microscope, real-time denoising has been achieved with our Matlab GUI of DeepCAD-RT (tested on a Windows desktop with Intel i9 CPU and 128 GB RAM).

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/GUI2.png?raw=true" width="950" align="middle"></center> 



### 4.1 Required environment

- Windows 10
- CUDA 11.0
- CUDNN 8.0.5
- Matlab 2018a (or newer version)
- Visual Studio 2017

### 4.2 File description

`deepcad_trt.m`: Matlab script that calls fast processing and tiff saving function programmed in C++

`deepcad_trt_nosave.m`: Matlab script that calls fast processing function programmed in C++ and save tiff in Matlab

`realtime_core.m`: Realtime simulation in Matlab & C++ and save tiff

`./deepcad/+deepcadSession`: Realtime inference with data flow from ScanImage

`./results`: save result images

`./model`: save engine file

### 4.3 Instructions for use

#### Install

Download the `.exe` file in our [cloud disk](https://cloud.tsinghua.edu.cn/f/89410848303d40889078/).

#### Model preparation

Before using a model, you should first convert the pth model to an ONNX model, and then convert the ONNX model to an Engine file. When you change your GPU, the Engine file should be rebuilt.

   **pth model to ONNX model:**

1. Go to `DeepCAD-RT/DeepCAD_RT_pytorch/` directory and activate the `deepcadrt` environment [[Configuration tutorial for conda environment](#python-source-code)].  

   ```
   $ conda activate deepcadrt
   $ cd DeepCAD-RT/DeepCAD_RT_pytorch/
   ```

2. Run the `convert_pth_to_onnx.py`. Parameters in the following command can be modified as required.

   ```
   $ os.system('python convert_pth_to_onnx.py --patch_x 200 --patch_y 200 --patch_t 80 --denoise_model ModelForPytorch --GPU 0')
   
   @parameters
   --patch_x, --patch_y, --patch_t: patch size in three dimensions
   --GPU: specify the GPU used for file conversion
   --denoise_model: the folder containing the pre-trained models.
   ```

   The recommended patch size is 200x200x80 pixels. Put the pth model and yaml file in `./pth` path.  The default name of ONNX file name is the model file name.

   We provide a pre-trained ONNX model in `./model` . The patch size of `cal_mouse_mean_200_40_full.onnx` and `cal_mouse_mean_200_80_full.onnx` are  200x200x40 pixels and 200x200x80 pixels, respectively. The calcium imaging data used for training these model were captured by our customized two-photon microscope:

​    *Key imaging parameters of training data:*

- *30Hz sampling rate, 500x500 μm2 field of view, 490x490 pixels.*
- *The imaging depth is ranging from 40 to 180 um.*
- *The imaging power is ranging from 66 to 99 mW.*

**ONNX model to Engine file:**

3. Run the following command in Windows cmd.exe. Parameters can be modified as required: 

```
xxx\trtexec.exe --onnx=deepcad_200_80.onnx --explicitBatch --saveEngine=deepcad_fp16_200_80.engine --workspace=2000 --fp16

@parameters
--onnx: ONNX file name
--saveEngine: Engine file name
```



#### Realtime inference with ScanImage

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/GUI.png?raw=true" width="600" align="middle"></center> 

Matlab configuration:

1. Open Matlab.

2. Change file path to xxx/DeepCAD-RT-v2.

3. Configure the environment:

   ```
   mex -setup C++
   
   installDeepCADRT
   ```

4. Open ScanImage and DeepCAD_RT GUI:

   ```
   scanimage
   
   DeepCAD_RT 
   ```

5. Set the parameters in GUI:

   `Engine file`: The path of the Engine file. Click `...` to open the file browser and choose the path.

   `Save path`:The path to save denoised images. Click `...` to open the file browser and choose the path

   `Frames number`: How many frames to acquire. It is equal to the value set in ScanImage. This parameter will update automatically when you click `Configure`. 

   <center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/scanimage_parameter.png?raw=true" width="250" align="middle"></center>

   **Attention: You should set the frame number before clicking `Configure`.**

   `Display setting`: 

   `Manual` mode: You can set the minimum and maximum intensity for image display.

   `Auto` mode: The contrast will be set automatically but slightly slower than `Manual` mode.

   `Advanced`: Advanced settings.

   `Patch size (x,y,t)`: The three parameters depend on the patch size you set when you convert Pytorch model to ONNX model.

   `Overlap factor`: The overlap factor between two adjacent patches. The recommended number is between 0.125 and 0.4. Larger overlap factor means better performance but lower inference speed.

   `Input batch size`: The number of frames per batch. The recommended value is between 50 and 100. It should be larger than the patch size in t dimension.

   `Overlap frames between batches`: The number of overlapping slices between two adjacent batches. The recommended value is between 5 and 10. More overlapping frames lead to better performance but lower inference speed.

6. After set all parameters, please click `Configure`. If you click `Configure` for the first time, the initialization program will execute automatically.

7. You can click `GRAB` in ScanImage and start imaging.

8. Before each imaging session, you should click  `Configure`.


### GUI demo video

<center><iframe width="800" height="500" src="https://www.youtube.com/embed/u1ejSaVvWiY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> </center>

