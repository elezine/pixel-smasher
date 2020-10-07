## We have forked from [BasicSR](https://github.com/xinntao/BasicSR)

For the original README document from BasicSR, refer to [README.md.txt](./README.md.txt).

## BasicSR

`TBD`

## Our changes

We have updated model tunings for high-resolution satellite data.  See [Datasets](#Datasets).

These new changes are authored by Ethan D. Kyzivat and Ekaterina Lezine.

## Image import and preprocessing workflow

0. Download images to Scenes folder
1. [plot_hists_serial.py](old_BasicSR/codes/utils/plot_hists_serial.py) > saves histogram for each scene
2. [Compute_mean_hist.py](old_BasicSR/codes/utils/Compute_mean_hist.py) > averages these histograms
3. [extract_subimgs_single.py](old_BasicSR/codes/scripts/extract_subimgs_single.py) > Divides satellite scenes into subsets
4. [rand_shuf.sh](old_BasicSR/codes/utils/rand_shuf.sh) > Randomly creates training and validation partitions
5. [generate_mod_LR_bic_parallel.py](old_BasicSR/codes/scripts/generate_mod_LR_bic_parallel.py) > Upscales and downscales subsets via a number of methods

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`
  
## Datasets
`TBD`

## Get Started
Please see [wiki](https://github.com/xinntao/BasicSR/wiki/Training-and-Testing) for the basic usage, *i.e.,* training and testing.
