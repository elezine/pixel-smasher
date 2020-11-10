import numpy as np
from multiprocessing import Pool
import multiprocessing
import rasterio as rio
import glob as gl
import os
import sys

georef_path = '/data_dir/Scenes-shield-gt-subsets/20170710_181144_1034_3B_AnalyticMS_SR_s0001.tif'

georef_img_rio = rio.open(georef_path)
profile = georef_img_rio.profile

print(profile)