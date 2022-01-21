---
layout: page
title: DeepCAD-RT tutorial
---

<img src="https://github.com/STAR-811/Deepcad-RT-page/blob/master/images/logo.PNG?raw=true" width="800" align="middle">





## Content

- [Directory structure](#directory-structure)
- [Pytorch code](#pytorch-code)
- [Matlab GUI  for DeepCAD-RT (HA)](#matlab-gui--for-deepcad-rt-ha)

## Directory structure

```
DeepCAD-RT
|---DeepCAD_RT_pytorch #Pytorch implementation of DeepCAD-RT#
|---|---demo_train_pipeline.py
|---|---demo_test_pipeline.py
|---|---transfer_pth_to_onnx.py
|---|---deepcad
|---|---|---__init__.py
|---|---|---utils.py
|---|---|---network.py
|---|---|---model_3DUnet.py
|---|---|---data_process.py
|---|---|---buildingblocks.py
|---|---|---test_collection.py
|---|---|---train_collection.py
|---|---|---movie_display.py
|---|---notebooks
|---|---|---demo_train_pipeline.ipynb
|---|---|---demo_test_pipeline.ipynb
|---|---|---DeepCAD_RT_demo_colab.ipynb
|---|---datasets
|---|---|---DataForPytorch # project_name #
|---|---|---|---data.tif
|---|---pth
|---|---|---ModelForPytorch
|---|---|---|---model.pth
|---|---|---|---model.yaml
|---|---results
|---|---|--- # test results#
|---DeepCAD_RT_HA #Matlab GUI of DeepCAD-RT#
```

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

To achieve real-time denoising during imaging process, DeepCAD-RT is implemented on GPU with Nvidia TensorRT and delicately-designed time sequence to further accelerate the inference speed and decrease memory cost. We developed a user-friendly Matlab GUI for DeepCAD-RT, which is easy to install and convenient to use (has been tested on a Windows desktop with Intel i9 CPU and 128G RAM).  **Tutorials** on installing and using the GUI has been moved to [**this page**](https://github.com/STAR-811/DeepCAD-RT/tree/main/DeepCAD_RT_HA).  

<div style="align: center">

​              <img src="https://github.com/STAR-811/Deepcad-RT-page/blob/master/images/GUI.png?raw=true" width="600" align="middle">   


