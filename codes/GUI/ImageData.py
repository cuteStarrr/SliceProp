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

def normalization(output):
    new_mask_image = np.zeros([output.shape[0],output.shape[1],output.shape[2]], dtype = np.float32)
    for i in range(output.shape[0]):           
        f_mask = output[i,:,:]
        zero = np.zeros_like(f_mask)
        one  = np.ones_like(f_mask)
        new_mask_image[i,:,:]= np.where(f_mask> 0.5, one, zero)
    return new_mask_image


class ImageData(object):
    def __init__(self):
        # for data
        self.penthickness = 2
        self.add_color = (0, 0, 255)
        self.remove_color = (0, 255, 0)
        self.isAdd = 1
        self.remove_anotation_flag = 0
        """
        add_seed的坐标对应方式与原始图像，也就是h5文件一致
        """
        self.add_seed = [] # height, width, depth
        self.remove_seed = [] # height, width, depth
        self.depth_current = 0
        self.crop_size = 96
        self.expand_size = (1024, 256, 256) # depth, height, width

        # for image

    def readH5(self, path):
        self.file_image = h5py.File(path, 'r')
        self.image_src = (self.file_image['image'])[()]
        self.image_label = (self.file_image['label'])[()]
        self.image_height, self.image_width, self.image_depth = self.image_src.shape
        """
        得到的event是x和y,也就是先在哪列，再在哪行
        """
        self.anotation_output = np.zeros((self.image_depth, self.image_height, self.image_width, 3), dtype=np.uint8)

    def getBGRImage(self, gray_image):
        gray_image = (gray_image * 255).astype(np.uint8)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    def getImage2show(self):
        return cv2.addWeighted(self.getBGRImage(self.image_src[:, :, self.depth_current]), 0.9, self.anotation_output[self.depth_current], 0.7, 0.7)

    def getImage2np(self, path):
        suffix = path.split(".")[-1]
        if suffix == 'h5':
            self.readH5(path)

    def anotate(self, x, y, isadd, depth = -1):
        if depth == -1:
            depth = self.depth_current
        
        if isadd:
            if not self.add_seed.__contains__((y, x, depth)):
                # print("add seed")
                self.add_seed.append((y, x, depth))
                cv2.rectangle(self.anotation_output[depth], (x - 1, y - 1), (x + 1, y + 1), self.add_color, 1) # 这里的颜色是GBR，要画的点可以变为(x, y), (x, y)，目前从肉眼来看区别不大
        else:
            if not self.remove_seed.__contains__((y, x, depth)):
                self.remove_seed.append((y, x, depth))
                self.remove_anotation_flag = 1
                cv2.rectangle(self.anotation_output[depth], (x - 1, y - 1), (x + 1, y + 1), self.remove_color, 1)


    def anotation_center_point(self, seeds):
        height, width, depth = 0., 0., 0.
        for seed in seeds:
            height = height + seed[0]
            width = width + seed[1]
            depth = depth + seed[2]
        """
        注意，排列需要按照depth, height, width
        """
        center = (int(round(depth / len(seeds))), int(round(height / len(seeds))), int(round(width / len(seeds))))
        left_up = []
        left_up.append(center[0] - self.crop_size // 2)
        left_up.append(center[1] - self.crop_size // 2)
        left_up.append(center[2] - self.crop_size // 2)
        return center, tuple(left_up)

    def expandLeftUp(self):
        phl = (self.expand_size[1] - self.image_height) // 2
        pwl = (self.expand_size[2] - self.image_width) // 2
        pdl = (self.expand_size[0] - self.image_depth) // 2

        return (pdl, phl, pwl)
        

    def class2Anotation(self, output):
        output = normalization(output)
        anotation = np.zeros((output.shape[0],output.shape[1],output.shape[2], 3), dtype = np.uint8)
        for i in range(output.shape[0]):
            tmp = self.getBGRImage(output[i])
            tmp = np.where(tmp == [0, 0, 0], [0, 0, 0], list(self.add_color))
            anotation[i, :, :, :] = tmp

        return anotation

    def recoverFromCrop(self, output, left_up_point):
        output = output[-min(0, left_up_point[0]):self.crop_size + min(0, self.image_depth - self.crop_size - left_up_point[0]), -min(0, left_up_point[1]):self.crop_size + min(0, self.image_height - self.crop_size - left_up_point[1]), -min(0, left_up_point[2]):self.crop_size + min(0, self.image_width - self.crop_size - left_up_point[2])]
        output = np.pad(output, [(max(left_up_point[0], 0), max(self.image_depth - self.crop_size - left_up_point[0], 0)), (max(0, left_up_point[1]), max(self.image_height - self.crop_size - left_up_point[1], 0)), (max(0, left_up_point[2]), max(self.image_width - self.crop_size - left_up_point[2], 0))], mode='constant', constant_values=0)
        return output

    def recoverFromExpand(self, output, left_up_point):
        output = output[left_up_point[0]:left_up_point[0] + self.image_depth, left_up_point[1]:left_up_point[1] + self.image_height, left_up_point[2]:self.output_size[2] + self.image_width]
        return output

    def combineAnotationOutput(self, output):
        for i in range(output.shape[0]):
            tmp1 = output[i]
            tmp2 = self.anotation_output[i]
            tmp = np.where(tmp1 == [0, 0, 0], tmp2, tmp1)
            self.anotation_output[i, :, :, :] = tmp


    def init_segment(self):
        """
        for crop image
        """
        net = UNet3D()
        model_weight_path = r'./addTogether/UNet3D_1.pth'
        net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
        center_point, left_up_point = self.anotation_center_point(self.add_seed) #depth, height, width
        print(center_point)
        transform = Compose([
                    CenterCrop(center_point, (self.crop_size, self.crop_size, self.crop_size)),
                    ToTensor()
                ])

        samples = (self.image_src).transpose((2,0,1)), (self.image_label).transpose((2,0,1))
        tr_samples = transform(samples)
        image_data, image_label = tr_samples
        print(image_data.shape)
        net.eval()
        output = net(image_data.unsqueeze(0).unsqueeze(0).float())
        output = output.squeeze(0).squeeze(0).detach().numpy() # output: depth, height, width
        output = self.recoverFromCrop(output, left_up_point)
        output = self.class2Anotation(output) # 得到了anotation矩阵
        self.combineAnotationOutput(output)
        self.add_seed = []
        # """
        # for expanded image
        # """
        # net = UNet3D()
        # model_weight_path = r'./addTogether/UNet3D_1.pth'
        # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
        # # center_point, left_up_point = self.anotation_center_point(self.add_seed) #depth, height, width
        # # print(center_point)
        # left_up_point = self.expandLeftUp()
        # transform = Compose([
        #             CenterExpand(self.expand_size),
        #             ToTensor_expand()
        #         ])

        # sample = (self.image_src).transpose((2,0,1))
        # image_data = transform(sample)
        
        # # print(type(image_data))
        # net.eval()
        # # input = image_data.unsqueeze(0).unsqueeze(0).float()
        # # print(input.shape)
        # output = net(image_data.unsqueeze(0).unsqueeze(0).float())
        # output = output.squeeze(0).squeeze(0).detach().numpy() # output: depth, height, width
        # output = self.recoverFromExpand(output, left_up_point)
        # output = self.class2Anotation(output) # 得到了anotation矩阵
        
        # return output
        
        

    def Clear(self):
        #清空画板
        print("Clear")