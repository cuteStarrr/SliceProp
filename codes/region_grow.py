import numpy as np
import os
import cv2
import h5py
import SimpleITK as sitk
from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries
import random
from interact_dataset import get_right_seeds

def seeds2map(seeds, map_shape):
    seeds_map = np.zeros(map_shape)
    for i in range(seeds.shape[0]):
        seeds_map[seeds[i, 0], seeds[i, 1]] = 1

    return np.uint8(seeds_map)

def delete_black_hole(seedsmap):
    src = np.uint8(np.where(seedsmap == 1, 0, 1))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8, ltype=None)
    # print("num_labels", num_labels)
    for i in range(1, num_labels):
        mask = labels == i             #这一步是通过labels确定区域位置，让labels信息赋给mask数组，再用mask数组做img数组的索引
        if stats[i][4] < 10:         #300是面积 可以随便调
            src[mask] = 1
        else:
            src[mask] = 0
    #num = np.sum(src > 0)
    src = seedsmap + src
    # print(num)
    return src



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
    return np.uint8(delete_black_hole(seeds_map))
    # return np.uint8(seeds_map)


if __name__ == '__main__':
    print("region grow")
    # image_path = r'D:\Lab\Aortic_Segmentation\training_files\bothkinds_masks\transform_sobel_scribble\validate_2_transform_sobel_scribble_loss_11_6.h5'
    # file_image = h5py.File(image_path, 'r')
    # image_data = (file_image['image'])[()]
    # image_label = (file_image['label'])[()]
    # test_image = image_data[:,:,16]
    # test_label = image_label[:,:,16]
    # test_label = np.where(test_label > 1, 0, test_label)
    # test_label = np.uint8(test_label)
    # _, seeds = get_right_seeds(test_label, test_image, test_image, 5, 0.2)
    # print(seeds.shape)
    # seeds_map = seeds2map(seeds, test_label.shape)
    # output, num_tmp = region_grow(test_image, seeds, 2)

    # print(num_tmp)