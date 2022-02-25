---
layout: page
title: About
---

### [GitHub](https://github.com/cabooster/DeepCAD-RT) | [Paper](https://www.nature.com/articles/s41592-021-01225-0)

## Content

- [Introduction](#introduction)
- [Performance](#performance)
- [Citation](#citation)

## Introduction

### Background

**Among the challenges of fluorescence microscopy, poor imaging signal-to-noise ratio (SNR) caused by limited photon budget lingeringly stands in the central position.** Fluorescence microscopy is inherently sensitive to detection noise because the photon flux in fluorescence imaging is usually three orders of magnitude lower than that in photography. 

Considering the detection physics, three sources mainly contribute to the detection noise in fluorescence imaging, namely **the dark noise, the photon shot noise, and the readout noise**. Among them, the dark noise and the photon noise follow a Poisson distribution and the readout noise follows a Gaussian distribution. Hence, the detection noise of fluorescence microscopy follows a Mixed Poisson-Gaussian (MPG) distribution. Benefiting from advanced manufacturing technologies of integrated circuits and high-performance sensor cooling methods, the dark noise and the readout noise are relatively low. **Thus, the photon shot noise plays the dominant role in the MPG noise of fluorescence microscopy.**

The causes of this photon-limited challenge are as follows:

- The **low photon yield** of fluorescent indicators and their low concentration in labeled cells result in the lack of photons at the source. 

- Although using higher excitation power is a straightforward way to increase fluorescence photons, living systems are too fragile to tolerate high excitation dosage. Extensive experiments have shown that illumination-induced **photobleaching, phototoxicity, and tissue heating** will disturb crucial cellular processes including cell proliferation, migration, vesicle release, neuronal firing, etc. 
- Recording fast biological processes necessitates high imaging speed and the **short dwell time** further intensifies the shortage of photons. 
- The **quantum nature of photons** makes the stochasticity (shot noise<sup>*</sup>) of optical measurements inevitable. The intensity detected by photoelectric sensors follows a Poisson distribution parameterized with the exact photon count. In fluorescence imaging, detection noise dominated by photon shot noise exacerbates the measurement uncertainty and obstructs the veritable visualization of underlying structures, potentially altering morphological and functional interpretations that follow. 

**In optics, shot noise describes fluctuations in the number of photons detected as they occur independently of each other. There are other noise mechanisms in optical signals that often make the contribution of shot noise appear negligible. When these noises are not present, optical detection is said to be 'photon noise limited' because only shot noise is left.*



<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/deepcad6.png?raw=true" width="1000" align="middle" /></center>

To capture enough photons for satisfactory imaging sensitivity, researchers have to sacrifice imaging speed, resolution, and even sample health.



### Our Contribution

...





## Performance

<center><h3>1. DeepCAD-RT massively improves the imaging SNR of neuronal population recordings in the zebrafish brain</h3></center>

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/gallery_zebra.png?raw=true" width="700" align="middle"></center>

<center><h3>2. DeepCAD-RT reveals the 3D migration of neutrophils in vivo after acute brain injury</h3></center>

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/gallery_NP.png?raw=true" width="850" align="middle"></center>

<center><h3>3. DeepCAD-RT reveals the ATP (Adenosine 5’-triphosphate) dynamics of astrocytes in 3D after laser-induced brain injury</h3></center>

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/gallery_ATP.png?raw=true" width="850" align="middle"></center>



<center>More demo images and videos are demonstrated on <a href='https://cabooster.github.io/DeepCAD-RT/Gallery/'>Gallery</a></center>







## Citation

If you use this code please cite the companion paper where the original method appeared: 

Li, X., Zhang, G., Wu, J. et al. Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising. Nat Methods (2021). [https://doi.org/10.1038/s41592-021-01225-0](https://www.nature.com/articles/s41592-021-01225-0)



