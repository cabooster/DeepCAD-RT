---
layout: page
title: About
---

### [GitHub](https://github.com/cabooster/DeepCAD-RT) | [Paper](https://www.nature.com/articles/s41592-021-01225-0)

## Content

- [Overview](#overview)
- [Performance](#performance)
- [Citation](#citation)

## Overview

<center><img src="https://github.com/STAR-811/Deepcad-RT-page/blob/master/images/schematic.png?raw=true" width="600" align="middle" /></center>

Calcium imaging is inherently susceptible to detection noise especially when imaging with high frame rate or under low excitation dosage. However, calcium transients are highly dynamic, non-repetitive activities and a firing pattern cannot be captured twice. Clean images for supervised training of deep neural networks are not accessible. Here, we present DeepCAD, a **deep** self-supervised learning-based method for **ca**lcium imaging **d**enoising. Using our method, detection noise can be effectively removed and the accuracy of neuron extraction and spike inference can be highly improved.

DeepCAD is based on the insight that a deep learning network for image denoising can achieve satisfactory convergence even the target image used for training is another corrupted sampling of the same scene [[paper link]](https://arxiv.org/abs/1803.04189). We explored the temporal redundancy of calcium imaging and found that any two consecutive frames can be regarded as two independent samplings of the same underlying firing pattern. A single low-SNR stack is sufficient to be a complete training set for DeepCAD. Furthermore, to boost its performance on 3D temporal stacks, the input and output data are designed to be 3D volumes rather than 2D frames to fully incorporate the abundant information along time axis.

For more details, please see the companion paper where the method first appeared: 
["*Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising*".](https://www.nature.com/articles/s41592-021-01225-0)



## Performance

<center><b>Universal denoising for calcium imaging in zebrafish.</b></center>

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/GN0IO7bGoGg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> </center><br>

<center><b>Denoising performance of DeepCAD-RT of neutrophils in the mouse brain in vivo</b></center>

<iframe width="560" height="315" src="https://www.youtube.com/embed/eyLPVRcEGHs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe><br>

<center><b>Denoising performance of DeepCAD-RT on a recently developed genetically encoded ATP sensor</center></b>

<iframe width="560" height="315" src="https://www.youtube.com/embed/kSMYJgE4M54" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe><br>

More demo videos are demonstrated on [Gallery](https://cabooster.github.io/DeepCAD-RT/Gallery/).

## Citation

If you use this code please cite the companion paper where the original method appeared: 

Li, X., Zhang, G., Wu, J. et al. Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising. Nat Methods (2021). [https://doi.org/10.1038/s41592-021-01225-0](https://www.nature.com/articles/s41592-021-01225-0)



<p><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/thu.jpg?raw=true" width="80" align="middle"><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/thu-auto.jpg?raw=true" width="80" align="middle"></p>



