---
layout: page
title: Datasets
---

## Content

- [Overview](#overview)
- [Download links](#download-links)
- [Citation](#citation)

## Overview

The data used for training and validation of DeepCAD-RT are made publicly available here. These data were captured by our customized two-photon microscope with two strictly synchronized detection path. The signal intensity of the high-SNR path is 10-fold higher than that of the low-SNR path. We provided 8 groups of recordings with various imaging depths, excitation power, and structures. All data are listed in the table below. You can download these data directly by clicking the `hyperlinks` appended in the 'AMP' column. [Fiji](https://imagej.net/software/fiji/downloads) can be used to open these image stacks.

## Download links

| No.  | Structures                                                | FOV (V×H)<sup>a</sup> | Frame rate | Imaging depth<sup>b</sup> |                      Power<sup>c</sup>                       | AMP<sup>d</sup> |
| :--: | --------------------------------------------------------- | --------------------- | :--------: | :-----------------------: | :----------------------------------------------------------: | :-------------: |
|  1   | Mouse dendritic spines                                    | 550×575 μm            |   30 Hz    |           40 μm           |                            50 mW                             |  [1]()/[10]()   |
|  2   | Zebrafish telencephalic neurons (local region)            | 550×575 μm            |   30 Hz    |          100 μm           |                              mW                              |  [1]()/[10]()   |
|  3   | Zebrafish telencephalic neurons (multiple regions)        | 550×575 μm            |   30 Hz    |          120 μm           |                              mW                              |  [1]()/[10]()   |
|  4   | Drosophila mushroom body                                  | 550×575 μm            |   30 Hz    |          140 μm           |                              mW                              |  [1]()/[10]()   |
|  5   | Mouse brain neutrophils                                   | 550×575 μm            |   30 Hz    |          160 μm           |                           0.25 mW                            |  [1]()/[10]()   |
|  6   | Mouse brain neutrophils (3D imaging)                      | 550×575 μm            |   30 Hz    | 15-45 μm (2 μm per step)  |                           0.07 mW                            |  [1]()/[10]()   |
|  7   | Extracellular ATP release in the mouse brain              | 550×575 μm            |   15 Hz    |           20 μm           |                           0.65 mW                            |  [1]()/[10]()   |
|  8   | Extracellular ATP release in the mouse brain (3D imaging) | 550×575 μm            |   30 Hz    | 10-70 μm (2 μm per step)  |                            0.1 mW                            |  [1]()/[10]()   |
|  9   |                                                           | 550×575 μm            |   30 Hz    |          150 μm           | [66](https://cloud.tsinghua.edu.cn/f/6be0ae5bfd2c439aa96d/?dl=1)/[99](https://cloud.tsinghua.edu.cn/f/d938acf4472841cc9d7c/?dl=1) mW |       10        |
|  10  |                                                           | 550×575 μm            |   30 Hz    |          170 μm           | [99](https://cloud.tsinghua.edu.cn/f/f94e5f874fcf428b81f3/?dl=1) mW |       10        |
|  11  |                                                           | 550×575 μm            |   30 Hz    |           80 μm           | [66](https://cloud.tsinghua.edu.cn/f/427de2eba72348d28a8e/?dl=1)/[99](https://cloud.tsinghua.edu.cn/f/5255b5709dec498783b9/?dl=1) mW |       10        |
|  12  |                                                           | 550×575 μm            |   30 Hz    |          110 μm           | [66](https://cloud.tsinghua.edu.cn/f/8b1a56b4e13c43999697/?dl=1)/[99](https://cloud.tsinghua.edu.cn/f/f18c6fc9f6e745a4a26a/?dl=1) mW |       10        |
|  13  |                                                           | 550×575 μm            |   30 Hz    |          185 μm           | [99](https://cloud.tsinghua.edu.cn/f/9dac2e30cf604809a833/?dl=1)/[132](https://cloud.tsinghua.edu.cn/f/d5550a6041a94b6282ca/?dl=1) mW |       10        |
|  14  |                                                           | 550×575 μm            |   30 Hz    |          210 μm           | [99](https://cloud.tsinghua.edu.cn/f/fda8bc14755a4f14b4ef/?dl=1)/[132](https://cloud.tsinghua.edu.cn/f/7a9e1cd0b1ab4effa02c/?dl=1) mW |       10        |

```
a.	FOV: field-of-view; V: vertical length; H: horizontal length.
b.	Depth: imaging depth below the pia mater.
c.	Two different excitation powers were used in each experiment for data diversity.
d.	AMP: the amplifier gain of the two PMTs.
```

## Citation

If you use our datasets please cite the companion paper: 

Li, X., Zhang, G., Wu, J. et al. Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising. Nat Methods (2021). [https://doi.org/10.1038/s41592-021-01225-0](https://www.nature.com/articles/s41592-021-01225-0)
