---
layout: page
title: Datasets
---

The data used for training and validation of DeepCAD-RT are made publicly available here. These data were captured by a standard two-photon microscope with multi-color detection capability and a customized two-photon microscope with two strictly synchronized detection paths. The signal intensity of the high-SNR path is 10-fold higher than that of the low-SNR path. We provided 11 groups of recordings, including the synthetic calcium imaging data, recordings of *in vivo* calcium dynamics in the brains of zebrafish and *Drosophila*,  as well as volumetric imaging in the mouse brain. All data are listed in the table below. You can download these data directly by clicking the `hyperlinks` appended in the 'Title' column. 

## Download links

| No.  |                            Title                             |      Events       |  Pixel size  | Frame/volume rate | Imaging Depth<sup>a</sup> | Data size |     Comments      |
| :--: | :----------------------------------------------------------: | :---------------: | :----------: | :---------------: | :-----------------------: | :-------: | :---------------: |
|  1   | [Synthetic calcium imaging data](https://doi.org/10.5281/zenodo.6254739) | Calcium transient | 1.020 μm/pxl |       30 Hz       |          200 μm           |  29.8 GB  | Low-SNR/high-SNR  |
|  2   |                    Mouse dendritic spines                    | Calcium transient | 0.155 μm/pxl |       30 Hz       |           40 μm           |  21.7 GB  | Low-SNR/high-SNR  |
|  3   |               Zebrafish telencephalic neurons                | Calcium transient | 0.254 μm/pxl |       30 Hz       |            ——             |  6.32 GB  | Low-SNR/high-SNR  |
|  4   |               Zebrafish multiple brain regions               | Calcium transient | 0.873 μm/pxl |       15 Hz       |            ——             |  7.18 GB  | Low-SNR/high-SNR  |
|  5   |                   Drosophila mushroom body                   | Calcium transient | 0.254 μm/pxl |       30 Hz       |            ——             |  11.1 GB  | Low-SNR/high-SNR  |
|  6   |                   Mouse brain neutrophils                    |  Cell migration   | 0.349 μm/pxl |       10 Hz       |           30 μm           |  11.8 GB  | Low-SNR/high-SNR  |
|  7   |                 Mouse brain neutrophils (3D)                 |  Cell migration   | 0.310 μm/pxl |       2 Hz        |  15-45 μm (2 μm/ plane)   |  27.4 GB  | Low-SNR, 2 colors |
|  8   |                ATP release in the mouse brain                |   ATP dynamics    | 0.465 μm/pxl |       15 Hz       |           20 μm           |  6.01 GB  | Low-SNR/high-SNR  |
|  9   |             ATP release in the mouse brain (3D)              |   ATP dynamics    | 0.698 μm/pxl |       1 Hz        |   10-70 μm (2 μm/plane)   |  46.7 GB  |      Low-SNR      |
|  10  |                        Mouse neurites                        | Calcium transient | 0.977 μm/pxl |       30 Hz       |         40-80 μm          |  23.5 GB  | Low-SNR/high-SNR  |
|  11  |                  Mouse neuronal populations                  | Calcium transient | 0.977 μm/pxl |       30 Hz       |         90-180 μm         |  49.9 GB  | Low-SNR/high-SNR  |

```
a.	Depth: imaging depth below the brain surface. Only for mouse experiments. 
```

## Citation

If you use our datasets please cite the companion paper: 

Li, X., Zhang, G., Wu, J. et al. Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising. Nat Methods (2021). [https://doi.org/10.1038/s41592-021-01225-0](https://www.nature.com/articles/s41592-021-01225-0)