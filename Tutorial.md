---
layout: page
title: DeepCAD-RT tutorial
---

<img src="https://github.com/STAR-811/Deepcad-RT-page/blob/master/images/logo.PNG?raw=true" width="800" align="middle">





## Content

- [Python code](#python-code)
- [Demo notebooks](#python-code)
- [Matlab implementation for real-time processing](#matlab-gui)

## Pytorch code

### Our environment 

* Ubuntu 16.04 
* Python 3.6
* Pytorch 1.8.0
* NVIDIA GPU (GeForce RTX 3090) + CUDA (11.1)

### Environment configuration

* Create a virtual environment, install Pytorch and DeepCAD package. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from https://pytorch.org/get-started/previous-versions/.

```
$ conda create -n deepcadrt python=3.6
$ conda activate deepcadrt
$ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install deepcad
```

### Download the source code

```
$ git clone git://github.com/cabooster/DeepCAD-RT
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/
```

### Demos

- Python file: 

  To try out the python file, please activate deepcadrt conda environment:

  ```
  $ conda activate deepcadrt
  $ cd DeepCAD-RT/DeepCAD_RT_pytorch/
  ```

  To  train your own DeepCAD-RT network, we recommend to start with the demo file `demo_train_pipeline.py`  in `DeepCAD_RT_pytorch` subfolder. You can try our demo files directly or edit some parameter appropriate to your hardware or data.

  **Example training**

  ```
  python demo_train_pipeline.py
  ```

  **Example test**

  ```
  python demo_test_pipeline.py
  ```

- Jupyter notebooks: 

  The notebooks provide a simple and friendly way to get into DeepCAD-RT. They are located in the `DeepCAD_RT_pytorch/notebooks`. To launch  the Jupyter notebooks:

  ```
  $ conda activate deepcadrt
  $ cd DeepCAD-RT/DeepCAD_RT_pytorch/notebooks
  $ jupyter notebook
  ```

- Colab notebooks: 

  You can also run Cellpose in google colab with a GPU: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/STAR-811/DeepCAD-RT/blob/main/DeepCAD_RT_pytorch/notebooks/DeepCAD_RT_demo_colab.ipynb)

## Matlab GUI

To achieve real-time denoising during imaging process, DeepCAD-RT is implemented on GPU with Nvidia TensorRT and delicately-designed time sequence to further accelerate the inference speed and decrease memory cost. We developed a user-friendly Matlab GUI for DeepCAD-RT , which is easy to install and convenient to use (has been tested on a Windows desktop with Intel i9 CPU and 128G RAM).  


​              <img src="https://github.com/STAR-811/Deepcad-RT-page/blob/master/images/GUI.png?raw=true" width="600" align="middle">   

### Required environment

- Windows 10
- Matlab 2018a (or newer version)
- CUDA 11.0
- CUDNN 8.0.5
- Visual Studio 2017

### File description

`deepcad_trt.m`: Fast processing in matlab & C++ and save tiff

`realtime_core.m`: Realtime simulation in matlab & C++ and save tiff

`./deepcad/+deepcadSession`: Realtime inference with data flow from ScanImage

`./results`: save result images

`./model`: save engine file

### Instructions for use

#### Install

Download exe file in our [cloud disk](https://cloud.tsinghua.edu.cn/f/ceebcebded4249c69540/).

#### Model preparation

Before inference, you should transfer pth model to ONNX model, then transfer ONNX model to Engine file. When you change your GPU, the Engine file should be rebuilt.

   **pth model to ONNX model:**

1. Go to `DeepCAD-RT/DeepCAD_RT_pytorch/` directory and activate `deepcadrt` conda environment. 

   ```
   $ conda activate deepcadrt
   $ cd DeepCAD-RT/DeepCAD_RT_pytorch/
   ```

2. Run the `transfer_pth_to_onnx.py`.  Parameters in following command can be modified as required.

   ```
   $ os.system('python transfer_pth_to_onnx.py --patch_x 200 --patch_y 200 --patch_t 80 --denoise_model ModelForPytorch --GPU 0')
   
   @parameters
   --patch_x, --patch_y, --patch_t: patch size in three dimensions
   --GPU: specify the GPU used for conversion
   --denoise_model: the folder containing the pre-trained models.
   ```

   The recommended patch size is 200 × 200 × 80 pixels, which can achieve the optimal performance. Put the pth model and yaml file in `./pth` path.  The default name of ONNX file name is the model file name.

   We also provide pre-trained ONNX model, which can be found in `./model` .  The patch size of `cal_mouse_mean_200_40_full.onnx` and `cal_mouse_mean_200_80_full.onnx` are  200 × 200 × 40 and 200 × 200 × 80, respectively. The calcium imaging data used for training these model were captured by our customized two-photon microscope on mouse spines:

​    *Key imaging parameters of training data:*

- *30Hz sampling rate, 500x500 μm2 field of view, 490x490 pixels.*
- *The imaging depth is ranging from 40 to 180 um.*
- *The imaging power is ranging from 66 to 99 mW.*

**ONNX model to Engine file:**

3. Run following  cmd command in Windows10 system. Parameters can be modified as required: 

```
xxx\trtexec.exe --onnx=deepcad_200_80.onnx --explicitBatch --saveEngine=deepcad_fp16_200_80.engine --workspace=2000 --fp16

@parameters
--onnx: ONNX file name
--saveEngine: Engine file name
```



#### Realtime inference with ScanImage

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

5. Set the parameters in GUI

   `Engine file`: Engine file root path. Click `...` to browse file folder and choose the path.

   `Save path`:Denoised image file save path. Click `...`  to browse file folder and choose the path

   `Frames number`: Frame number of the noisy image you set in ScanImage interface. This parameter will update automatically when you click `Configure`. 

   <img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/scanimage_parameter.png?raw=true" width="250" align="middle">

   **Attention: You should set frames number before you click `Configure`.**

   `Display setting`: 

   `Manual` mode: You can design the minimum and maximum intensity value for noisy/denoised image display.

   `Auto` mode: The contrast of display image will be set automatically. It will be a little slower than `Manual` mode.

   `Advanced`: Advanced parameters setting.

   `Patch size (x,y,t)`: These three parameters depend on the patch size you set when convert Pytorch model to ONNX model.

   `Overlap factor`: The overlap factor between two adjacent patches. The recommend number is between 0.125 to 0.4. Larger overlap factor means better performance but lower inference speed.

   `Input batch size`: The number of frames per batch. The recommend number is between 50 to 100. It should be larger than patch size in t dimension.

   `Overlap frames between batches`: The number of overlap slice between neighboring batch. The recommend number is between 5 to 10. More overlap frames mean better performance but lower inference speed.

6. After set all parameters, click  `Configure`. When you click `Configure` for the first time, the initializer program will execute  automatically.

7. You can click `GRAB` in ScanImage and begin imaging.

8. Before every imaging process, you should click  `Configure`.

### Demo video

<iframe width="560" height="315" src="https://www.youtube.com/embed/u1ejSaVvWiY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> 

