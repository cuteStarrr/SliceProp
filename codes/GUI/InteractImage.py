"""
数据格式记录：
image_src(h5文件)：height，width，depth的三维数组，取值范围[0,1]
每次选取特定depth，则处理二维矩阵，height——行数，width——列数——直接视作2D图像

anotation_output(用来表示标注数据)，depth，height，width，red，green，blue三通道（因为QImage是RGB三通道显示的）

"""

import os
from collections import OrderedDict
from os.path import join as opj
from turtle import left, shape
from typing import Any
from xml.sax.handler import DTDHandler

import cv2
import matplotlib.pyplot as plt
import maxflow
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import color, measure
import h5py
from ..UNet_COPY import *
from ..interact_dataset import *


class InteractImage(object):
    def __init__(self, image_path):
        """
        add_seed的坐标对应方式与原始图像，也就是h5文件一致
        """
        self.TL_seeds = [] # height, width, depth
        self.FL_seeds = [] 
        self.background_seed = [] # height, width, depth
        file = h5py.File(image_path, 'r')

        self.image = (file['image'])[()]
        self.height, self.width, self.depth = self.image.shape
        self.depth_current = self.depth // 2
        self.prediction = np.zeros((self.depth, self.height, self.width, 3), dtype=np.uint8)

    def set_depth(self, depth):
        self.depth_current = depth

    def gray2BGRImage(self, gray_image):
        gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
        # gray_image = (gray_image * 255).astype(np.uint8)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    def getImage2show(self):
        return cv2.addWeighted(self.gray2BGRImage(self.image[:, :, self.depth_current]), 0.9, self.prediction[self.depth_current], 0.7, 0.7)
