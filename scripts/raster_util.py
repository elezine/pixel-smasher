import cv2
import mmcv
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
import glob as gl
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.merge import merge
import glob as gl
from osgeo import gdal,ogr,osr,gdalconst
from affine import Affine
from skimage import exposure
from skimage.util import img_as_ubyte
import fiona
import rasterio
import rasterio.mask
from rasterio.windows import Window
from rasterio.features import bounds
from rasterio.transform import rowcol
import math
import geopandas as gpd
from shapely.geometry import mapping
from shapely.geometry import box

'''
Most of these functions are hard-coded specifically for Landsat.
'''


def stack_tif(im_folder, save_path, band_order = [3,4,5]):
    '''
    Given a folder of landsat image bands, saves a single image with bands stacked.
    Band order is default 3,4,5, but you could change this. Currently only works for single digit bands, modify for others.
    I had to hard code the crs in here, it's the wkt string for EPSG:32612 because rasterio didn't recognize the landsat crs...
    '''
    ims = gl.glob(im_folder + '/*.tif')
    to_stack = []
    for band in band_order:
        to_stack.append(rio.open(ims[0][:-6] + 'B' + str(band) + '.tif').read(1))
    
    stacked = np.stack(to_stack,axis=2)
    stacked = np.rollaxis(stacked, axis=2)
    profile = rio.open(ims[0][:-6] + 'B' + str(band_order[0]) + '.tif').profile

    profile.update({'count':3})
    profile.update({'crs':'PROJCS["WGS_1984_UTM_Zone_12N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-111],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["Meter",1]]'})
    
    with rio.open(save_path, "w", **profile) as dest:
        dest.write(stacked)

        
def chunk_tif(im_path, chunk_size, save_path):
    from rasterio.mask import mask
    from shapely import geometry
    '''
    Saves square chunks of larger landsat image as geotiffs, preserving spatial information.
    '''
    
    def getTileGeom(transform, x, y, squareDim):
        corner1 = (x, y) * transform
        corner2 = (x + squareDim, y + squareDim) * transform
        return geometry.box(corner1[0], corner1[1],
                            corner2[0], corner2[1])
    
    def getCellFromGeom(img, geom, filename, count):
        crop, cropTransform = mask(img, [geom], crop=True)
        if len(np.unique(crop)) > 1:
            writeImageAsGeoTIFF(crop,
                                cropTransform,
                                img.meta,
                                img.crs,
                                filename+"_"+str(count))
        
    def writeImageAsGeoTIFF(img, transform, metadata, crs, filename):
        metadata.update({"driver":"GTiff",
                         "height":img.shape[1],
                         "width":img.shape[2],
                         "transform": transform,
                         "crs": crs})
        metadata.update({'crs':'PROJCS["WGS_1984_UTM_Zone_12N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-111],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["Meter",1]]'})
        with rio.open(filename+".tif", "w", **metadata) as dest:
            dest.write(img)
    
    img = rio.open(im_path)
    
    numberOfCellsWide = int(img.shape[1] // (chunk_size/2))
    numberOfCellsHigh = int(img.shape[0] // (chunk_size/2))
    x, y = 0, 0
    count = 0
    for hc in range(numberOfCellsHigh):
        y = hc * (chunk_size/2)
        for wc in range(numberOfCellsWide):
            x = wc * (chunk_size/2)
            try:
                geom = getTileGeom(img.transform, x, y, chunk_size)
                getCellFromGeom(img, geom, save_path + '/chunk', count)
                count = count + 1
            except:
                'outside geometry?'
                
def rescale_reflectance(img, btm_percentile=2, top_percentile=98, individual_band=True):
    '''Rescales each band in each image individualy (unless flag 'individual_band' is False), given input btm and top percentiles. Preserves input nodata value of 0.'''
    # Contrast stretching
    mask = np.sum(img, axis=2)==0
    btm_val, top_val = [None]*img.shape[2], [None]*img.shape[2] # init
    if individual_band==True:
        for i in range(img.shape[2]):
            btm_val[i] = np.percentile(img[:,:,i][img[:,:,i]>0], btm_percentile)
            top_val[i] = np.percentile(img[:,:,i][img[:,:,i]>0], top_percentile)
            img[:,:,i] = exposure.rescale_intensity(img[:,:,i], in_range=(btm_val[i], top_val[i]))
    else:
        btm_val = np.percentile(img[np.sum(img, 2)>0], btm_percentile)
        top_val = np.percentile(img[np.sum(img, 2)>0], top_percentile)
        img = exposure.rescale_intensity(img, in_range=(btm_val, top_val))
    img = img_as_ubyte(img) #(img/65535*255).astype(np.uint8) # 

            # preserve nodata value
    img[img<255]+=1
    img[mask] = 0 # set nodata==0
    
    return img


def save_as_pngs(folder_path, output_folder):
    '''
    converts tifs in folder to pngs and saves in output folder
    '''
    imgs = gl.glob(folder_path + '/*.tif')
    for img in imgs:
        im = rio.open(img).read()
        im = np.rollaxis(im,axis=0,start=3)
        
        im = rescale_reflectance(im, btm_percentile = 5, top_percentile = 95, individual_band = False)
        
        name = os.path.basename(img)[:-4]
        cv2.imwrite(output_folder + '//' + str(name) + '.png', im)

def make_tifs_pngs(folder_path, tif_folder, save_folder, new_res = 3.0):
    '''
    Turns pngs into tifs with a new resolution.
    '''
    pngs = gl.glob(folder_path + '/*.png')

    for png in pngs:
        name = os.path.basename(png)[:-17]
        tif = rio.open(tif_folder + '//' + name + '.tif')
        
        profile = tif.profile
        old_affine = profile['transform']
        new_affine = Affine(new_res, old_affine.b, old_affine.c, old_affine.d, -new_res, old_affine.f)
        profile.update({'transform':new_affine})
        profile.update({'dtype':'uint8'})
        profile.update({'width':480, 'height':480})
        
        with rio.open(save_folder + '//' + name + '.tif', "w", **profile) as dest:
            dest.write(rio.open(png).read())
            

def crop_tifs(folder_path, crop, no_data):
    '''
    crops all images in folder by crop amount of pixels on each side,
    replacing with no_data value. this overwrites the former files
    '''
    ims = gl.glob(folder_path + '/*.tif')
    
    for im in ims:
        img = rio.open(im)
        profile = img.profile
        img = img.read()
        
        img[:,0:crop,:] = no_data
        img[:,:,0:crop] = no_data
        img[:,-crop:,:] = no_data
        img[:,:,-crop:] = no_data
        
        profile.update({'nodata' : no_data})
       
        with rio.open(im, "w", **profile) as dest:
            dest.write(img)
            
def merge_tifs(folder_path, output_path, og_landsat_path):
    '''
    this merges all the tifs in a folder and saves them at output_path
    '''
    tifs = gl.glob(folder_path + '/*.tif')
    tif_list = []
    for tif in tifs:
        tif_list.append(rio.open(tif))
    
    img, transform = merge(tif_list)
    
    og_landsat = rio.open(og_landsat_path)

    profile = og_landsat.profile
    profile.update({'transform':transform})
    profile.update({'width':img.shape[2], 'height':img.shape[1]})
    profile.update({'dtype':'uint8'})

    with rio.open(output_path, "w", **profile) as dest:
        dest.write(img)