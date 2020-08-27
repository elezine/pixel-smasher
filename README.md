## We have forked from [BasicSR](https://github.com/xinntao/BasicSR)

For the original README document from BasicSR, refer to [README.md.txt](./README.md.txt).

## BasicSR

`TBD`

## Our changes

We have updated model tunings for high-resolution satellite data.  See [Datasets](#Datasets).

These new changes are authored by Ethan D. Kyzivat and Ekaterina Lezine.

## Image import and preprocessing workflow

1. Download images to Scenes folder
2. [extract_subimgs_single.py](codes/scripts/extract_subimgs_single.py) > Divides satellite scenes into subsets
3. [rand_shuf.sh](codes/utils/rand_shuf.sh) > Randomly creates training and validation partitions
4. [generate_mod_LR_bic_parallel.py](/codes/scripts/generate_mod_LR_bic_parallel.py) > Upscales and downscales subsets via a number of methods

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
