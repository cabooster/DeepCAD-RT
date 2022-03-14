# DeepCAD-RT: A versatile toolbox for microscopy imaging denoising

<img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/logo.PNG?raw=true" width="700" align="middle" />

### [Project page](https://cabooster.github.io/DeepCAD-RT/) | [Paper](https://www.nature.com/articles/s41592-021-01225-0)



## Overview

**Among the challenges of fluorescence microscopy, poor imaging signal-to-noise ratio (SNR) caused by limited photon budget lingeringly stands in the central position.** Fluorescence microscopy is inherently sensitive to detection noise because the photon flux in fluorescence imaging is far lower than that in photography. For almost all fluorescence imaging technologies, the inherent [**shot-noise limit**](https://cabooster.github.io/DeepCAD-RT/About/) determines the upper bound of imaging SNR and restricts the imaging resolution, speed, and sensitivity. To capture enough fluorescence photons for satisfactory signal-to-noise ratio (SNR), researchers have to sacrifice imaging speed, resolution, and even sample health.  

We present a versatile method **DeepCAD-RT** to denoise fluorescence images with rapid processing speed that can be incorporated with the microscope acquisition system to achieve real-time denoising. Our method is based on deep self-supervised learning and the original low-SNR data can be directly used for training convolutional networks, making it particularly advantageous in functional imaging where the sample is undergoing fast dynamics and capturing ground-truth data is hard or impossible. We have demonstrated extensive experiments including calcium imaging in mice, zebrafish, and flies, cell migration observations, and the imaging of a new genetically encoded ATP sensor, covering both 2D single-plane imaging and 3D volumetric imaging. **Qualitative and quantitative evaluations show that our method can substantially enhance fluorescence time-lapse imaging data and permit high-sensitivity imaging of biological dynamics beyond the shot-noise limit.**



For more details, please see the companion paper where the method first appeared: 
["*Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising, Nature Methods (2021)*"](https://www.nature.com/articles/s41592-021-01225-0).



<img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/schematic.png?raw=true" width="800" align="middle">



## Pytorch code

### Our environment 

* Ubuntu 16.04 
* Python 3.6
* Pytorch 1.8.0
* NVIDIA GPU (GeForce RTX 3090) + CUDA (11.1)

### Environment configuration

1. Create a virtual environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).

   ```
   $ conda create -n deepcadrt python=3.6
   $ conda activate deepcadrt
   $ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. We made a pip installable realease of DeepCAD [pypi](https://pypi.org/project/deepcad/). You can install it by simply entering following command:

   ```
   $ pip install deepcad
   ```

### Download the source code

```
$ git clone git://github.com/cabooster/DeepCAD-RT
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/
```

### Demos

To try out the python file, please activate `deepcadrt` conda environment:

```
$ conda activate deepcadrt
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/
```

**Example training**

To  train your own DeepCAD-RT network, we recommend to start with the demo file `demo_train_pipeline.py`  in `DeepCAD_RT_pytorch` subfolder. You can try our demo files directly or edit training parameters appropriate to your hardware and data. 

```
python demo_train_pipeline.py
```

**Example testing**

To test the denoising performance with pre-trained models, you can use our demo data and correspoding models or edit parameters to test your own model in the demo file `demo_test_pipeline.py` .

```
python demo_test_pipeline.py
```

### Jupyter notebook

The notebooks `demo_train_pipeline.ipynb` and `demo_test_pipeline.ipynb` provide a simple and friendly way to implement DeepCAD-RT. They are located in the `DeepCAD_RT_pytorch/notebooks`. Before you launch the Jupyter notebooks, please configure the `deepcadrt` environment following the instruction in [Environment configuration](#environment-configuration) . And then, you can try out the notebooks by typing following commands:

```
$ conda activate deepcadrt
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/notebooks
$ jupyter notebook
```

### Colab notebook

We also provide a cloud-based demo implemented with Google Colab. You can run DeepCAD in your browser using a cloud GPU without configuring the environment. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cabooster/DeepCAD-RT/blob/master/DeepCAD_RT_pytorch/notebooks/DeepCAD_RT_demo_colab.ipynb)

*This is a simple example with a slow rate because of the limited GPU performance offered by Colab. You can increase the `train_datasets_size` and `n_epochs` with a more powerful GPU, and training and testing time can be further shortened.*



## Matlab GUI

To achieve real-time denoising during imaging process, DeepCAD-RT is implemented on GPU with Nvidia TensorRT and delicately-designed time sequence to further accelerate the inference speed and decrease memory cost. We developed a user-friendly Matlab GUI for DeepCAD-RT , which is easy to install and convenient to use (has been tested on a Windows desktop with Intel i9 CPU and 128G RAM).  **Tutorials** on installing and using the GUI has been moved to [**this page**](https://github.com/STAR-811/DeepCAD-RT/tree/master/DeepCAD_RT_GUI).  

<img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/GUI2.png?raw=true" width="950" align="middle">

## Results

**1. Universal denoising for calcium imaging in zebrafish.**

[![IMAGE ALT TEXT](https://github.com/cabooster/DeepCAD-RT/blob/page/images/sv3_video.png?raw=true.png)]( https://www.youtube.com/embed/GN0IO7bGoGg "Video Title")

**2. Denoising performance of DeepCAD-RT of neutrophils in the mouse brain in vivo.** 

[![IMAGE ALT TEXT](https://github.com/cabooster/DeepCAD-RT/blob/page/images/sv8_video.png?raw=true.png)]( https://www.youtube.com/embed/eyLPVRcEGHs "Video Title")

**3. Denoising performance of DeepCAD-RT on a recently developed genetically encoded ATP sensor.**

[![IMAGE ALT TEXT](https://github.com/cabooster/DeepCAD-RT/blob/page/images/sv10_video.png?raw=true.png)](https://www.youtube.com/embed/u1ejSaVvWiY "Video Title")

More demo videos are demonstrated on [our website](https://cabooster.github.io/DeepCAD-RT/Gallery/).

## Citation

If you use this code please cite the companion paper where the original method appeared: 

Li, X., Zhang, G., Wu, J. et al. Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising. Nat Methods 18, 1395–1400 (2021). [https://doi.org/10.1038/s41592-021-01225-0](https://www.nature.com/articles/s41592-021-01225-0)

```
@article{li2021reinforcing,
  title={Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising},
  author={Li, Xinyang and Zhang, Guoxun and Wu, Jiamin and Zhang, Yuanlong and Zhao, Zhifeng and Lin, Xing and Qiao, Hui and Xie, Hao and Wang, Haoqian and Fang, Lu and others},
  journal={Nature Methods},
  volume={18},
  number={11},
  pages={1395--1400},
  year={2021},
  publisher={Nature Publishing Group}
}
```



