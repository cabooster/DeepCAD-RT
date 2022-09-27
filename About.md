---
layout: page
title: About
---
<img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/logo.PNG?raw=true" width="700" align="right" />
### [GitHub](https://github.com/cabooster/DeepCAD-RT) | [Paper](https://www.nature.com/articles/s41587-022-01450-8)

## Content

- [Introduction](#introduction)
- [Results](#results)
- [Citation](#citation)

## Introduction

### Background

**Among the challenges of fluorescence microscopy, poor imaging signal-to-noise ratio (SNR) caused by limited photon budget lingeringly stands in the central position.** Fluorescence microscopy is inherently sensitive to detection noise because the photon flux in fluorescence imaging is much lower than that in photography. To capture enough fluorescence photons for satisfactory SNR, researchers have to sacrifice imaging speed, resolution, and even sample health. The causes of this photon-limited challenge are as follows:

- The **low photon yield** of fluorescent indicators and their **low concentration** in labeled cells result in the lack of photons at the source. 

- Although using higher excitation power is a straightforward way to increase fluorescence photons, living systems are too fragile to tolerate high excitation dosage. Extensive experiments have shown that illumination-induced **photobleaching, phototoxicity, and tissue heating** will disturb crucial cellular processes including cell proliferation, migration, vesicle release, neuronal firing, etc. 

- Recording fast biological processes necessitates high imaging speed and the **short dwell time** further intensifies the shortage of photons.
 
- The **quantum nature of photons** makes the stochasticity (shot noise<sup>*</sup>) of optical measurements inevitable. The intensity detected by photoelectric sensors follows a Poisson distribution parameterized with the exact photon count. In fluorescence imaging, detection noise dominated by photon shot noise exacerbates the measurement uncertainty and obstructs the veritable visualization of underlying structures, potentially altering morphological and functional interpretations that follow. 

***Shot noise**: In optics, shot noise describes the fluctuations of the number of photons detected due to their occurrence independent of each other. There are other mechanisms of noise in optical signals which often dwarf the contribution of shot noise. But when these noises are suppressed, optical detection is said to be 'photon noise limited' because only shot noise is left. [[Wikipedia]](https://en.wikipedia.org/wiki/Shot_noise#)*



<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/deepcad6.png?raw=true" width="1000" align="middle" /></center>


### Noise model

Considering the detection physics, three sources mainly contribute to the detection noise in fluorescence imaging, namely **the dark noise, the photon shot noise, and the readout noise**. Among them, the dark noise and the shot noise follow a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) and the readout noise follows a [Gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution). Hence, the detection noise of fluorescence microscopy follows a **Mixed Poisson-Gaussian (MPG) distribution** [[paper link]](https://ieeexplore.ieee.org/document/8327626). Benefiting from advanced manufacturing technologies of integrated circuits and high-performance sensor cooling methods, the dark noise and the readout noise are relatively low. Thus, **the photon shot noise plays the dominant role in fluorescence microscopy.**


### Our Contribution

We present a versatile method **DeepCAD-RT** to denoise fluorescence images with rapid processing speed that can be incorporated with the microscope acquisition system to achieve real-time denoising. Our method is based on deep self-supervised learning and the original low-SNR data can be directly used for training convolutional networks, making it particularly advantageous in functional imaging where the sample is undergoing fast dynamics and capturing ground-truth data is hard or impossible. We have demonstrated extensive experiments including calcium imaging in mice, zebrafish, and flies, cell migration observations, and the imaging of a new genetically encoded ATP sensor, covering both 2D single-plane imaging and 3D volumetric imaging. Qualitative and quantitative evaluations show that our method can substantially enhance fluorescence time-lapse imaging data and permit high-sensitivity imaging of biological dynamics beyond the shot-noise limit.



## Results

<center><h3>1. DeepCAD-RT massively improves the imaging SNR of neuronal population recordings in the zebrafish brain</h3></center>

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/gallery_zebra.png?raw=true" width="850" align="middle"></center>

<center><h3>2. DeepCAD-RT reveals the 3D migration of neutrophils in vivo after acute brain injury</h3></center>

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/gallery_NP.png?raw=true" width="850" align="middle"></center>

<center><h3>3. DeepCAD-RT reveals the ATP (Adenosine 5’-triphosphate) dynamics of astrocytes in 3D after laser-induced brain injury</h3></center>

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/gallery_ATP.png?raw=true" width="850" align="middle"></center>


<center>More demo images and videos are demonstrated in <a href='https://cabooster.github.io/DeepCAD-RT/Gallery/'>Gallery</a>. More details please refer to <a href='https://www.nature.com/articles/s41587-022-01450-8'>the companion paper</a></center>.


## Citation

If you use this code please cite the companion paper where the original method appeared: 

- Xinyang Li, Yixin Li, Yiliang Zhou, et al. Real-time denoising enables high-sensitivity fluorescence time-lapse imaging beyond the shot-noise limit. Nat Biotechnol (2022). [https://doi.org/10.1038/s41587-022-01450-8](https://www.nature.com/articles/s41587-022-01450-8)

- Xinyang Li, Guoxun Zhang, Jiamin Wu, et al. Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising. Nat Methods 18, 1395–1400 (2021). [https://doi.org/10.1038/s41592-021-01225-0](https://www.nature.com/articles/s41592-021-01225-0) 



