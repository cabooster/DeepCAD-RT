# Matlab implementation for real-time processing

To achieve real-time denoising, DeepCAD-RT was optimally deployed on GPU using TensorRT (Nvidia) for further acceleration and memory reduction. We also designed a sophisticated time schedule for multi-thread processing. Based on a two-photon microscope, real-time denoising has been achieved with our Matlab GUI of DeepCAD-RT (tested on a Windows desktop with Intel i9 CPU and 128 GB RAM).

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/GUI2.png?raw=true" width="950" align="middle"></center> 

## Contents

- [Required environment](#required-environment)
- [File description](#file-description)
- [Instructions for use](#instructions)
- [Demo video](#demo-video)

## Required environment

- Windows 10
- CUDA 11.0
- CUDNN 8.0.5
- Matlab 2018a (or newer version)
- Visual Studio 2017

## File description

`deepcad_trt.m`: Fast processing and save tiff in matlab & C++

`deepcad_trt_nosave.m`: Fast processing in C++ and save tiff in matlab

`realtime_core.m`: Realtime simulation in matlab & C++ and save tiff

`./deepcad/+deepcadSession`: Realtime inference with data flow from ScanImage

`./results`: save result images

`./model`: save engine file

## Instructions

### Install

Download the `.exe` file in our [cloud disk](https://cloud.tsinghua.edu.cn/f/89410848303d40889078/).

### Model preparation

Before inference, you should convert pth model to ONNX model, and then convert ONNX model to Engine file. When you change your GPU, the Engine file should be rebuilt.

   **pth model to ONNX model:**

1. Go to `DeepCAD-RT/DeepCAD_RT_pytorch/` directory and activate `deepcadrt` conda environment [[Configuration tutorial for conda environment](#python-source-code)].  

   ```
   $ conda activate deepcadrt
   $ cd DeepCAD-RT/DeepCAD_RT_pytorch/
   ```

2. Run the `convert_pth_to_onnx.py`.  Parameters in following command can be modified as required.

   ```
   $ os.system('python convert_pth_to_onnx.py --patch_x 200 --patch_y 200 --patch_t 80 --denoise_model ModelForPytorch --GPU 0')
   
   @parameters
   --patch_x, --patch_y, --patch_t: patch size in three dimensions
   --GPU: specify the GPU used for conversion
   --denoise_model: the folder containing the pre-trained models.
   ```

   The recommended patch size is 200 × 200 × 80 pixels, which can achieve the optimal performance. Put the pth model and yaml file in `./pth` path.  The default name of ONNX file name is the model file name.

   We also provide pre-trained ONNX model, which can be found in `./model` .  The patch size of `cal_mouse_mean_200_40_full.onnx` and `cal_mouse_mean_200_80_full.onnx` are  200 × 200 × 40 pixels and 200 × 200 × 80 pixels, respectively. The calcium imaging data used for training these model were captured by our customized two-photon microscope on mouse spines:

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



### Realtime inference with ScanImage

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

   `Engine file`: Engine file root path. Click `...` to browse file folder and choose the path.

   `Save path`:Denoised image file save path. Click `...`  to browse file folder and choose the path

   `Frames number`: Frame number of the noisy image you set in ScanImage interface. This parameter will update automatically when you click `Configure`. 

   <center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/scanimage_parameter.png?raw=true" width="250" align="middle"></center>

   **Attention: You should set frames number before you click `Configure`.**

   `Display setting`: 

   `Manual` mode: You can design the minimum and maximum intensity value for noisy/denoised image displayment.

   `Auto` mode: The contrast of display image will be set automatically. It will be a little slower than `Manual` mode.

   `Advanced`: Advanced parameters setting.

   `Patch size (x,y,t)`: These three parameters depend on the patch size you set when you convert Pytorch model to ONNX model.

   `Overlap factor`: The overlap factor between two adjacent patches. The recommended number is between 0.125 and 0.4. Larger overlap factor means better performance but lower inference speed.

   `Input batch size`: The number of frames per batch. The recommended number is between 50 and 100. It should be larger than patch size in t dimension.

   `Overlap frames between batches`: The number of overlap slice between neighboring batch. The recommended number is between 5 and 10. More overlap frames mean better performance but lower inference speed.

6. After set all parameters, please click  `Configure`. When you click `Configure` for the first time, the initializer program will execute automatically.

7. You can click `GRAB` in ScanImage and begin imaging.

8. Before every imaging process, you should click  `Configure`.

## Demo video



[![IMAGE ALT TEXT](../images/sv1_video.png)](https://www.youtube.com/embed/u1ejSaVvWiY "Video Title")

