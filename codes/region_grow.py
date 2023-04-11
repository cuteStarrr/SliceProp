import numpy as np
import os
import cv2
import h5py
import SimpleITK as sitk
from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries
import random

def seeds2map(seeds, map_shape):
    seeds_map = np.zeros(map_shape)
    for i in range(seeds.shape[0]):
        seeds_map[seeds[i, 0], seeds[i, 1]] = 1

    return np.uint8(seeds_map)



def region_grow(image, seeds, threshold = 3):
    """
    针对一个连通区域，看作正态分布，以2*threshold的范围进行region grow
    """
    ele = []
    for i in range(seeds.shape[0]):
        ele.append(image[seeds[i, 0], seeds[i, 1]])
    ele = np.array(ele)
    seeds_map = seeds2map(seeds, image.shape)
    flag = False

    while True:
        boundaries = find_boundaries(seeds_map, mode='outer').astype(np.uint8)
        coord = np.argwhere(boundaries > 0)
        for i in range(coord.shape[0]):
            if image[coord[i, 0], coord[i, 1]] >= ele.mean() - threshold * np.sqrt(ele.var()) and image[coord[i, 0], coord[i, 1]] <= ele.mean() + threshold * np.sqrt(ele.var()):
                seeds_map[coord[i, 0], coord[i, 1]] = 1
                ele = np.append(ele, image[coord[i, 0], coord[i, 1]])
                flag = True
        
        if not flag:
            break
        else:
            flag = False

    return seeds_map


if __name__ == '__main__':
    print("region grow")