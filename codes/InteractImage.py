"""
数据格式记录：
image_src(h5文件)：height，width，depth的三维数组，取值范围[0,1]
每次选取特定depth，则处理二维矩阵，height——行数，width——列数——直接视作2D图像

anotation_output(用来表示标注数据)，depth，height，width，red，green，blue三通道（因为QImage是RGB三通道显示的, opencv是BGR三通道显示的）

"""

import os
from collections import OrderedDict
from os.path import join as opj
from turtle import left, shape
from typing import Any
from xml.sax.handler import DTDHandler

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import color, measure
import h5py
from UNet_COPY import *
from interact_dataset import *
from train import accuracy_all_numpy
from test import get_prediction_all_bidirectional, get_network_input_all, get_prediction_all
from region_grow import *


"""
NEED TO DO:
1. self.prediction2anotation
2. background seeds
3. refinement
4. 初始分割的时候，更新每个depth对应的seeds
5. seeds clean需要修改，应该在得到seeds的时候就clean-即针对一个连通区域
"""


class InteractImage(object):
    def __init__(self, image_path):
        """
        add_seed的坐标对应方式与原始图像，也就是h5文件一致
        """
        self.FL_flag = False
        self.TL_flag = False
        self.background_flag = False
        self.TL_label = int(1)
        self.FL_label = int(2)
        self.background_label = int(0)
        self.TL_color = (0, 0, 255)
        self.FL_color = (255, 0, 0) # BGR
        self.background_color = (0, 255, 0)
        file = h5py.File(image_path, 'r')

        self.image = (file['image'])[()]
        self.height, self.width, self.depth = self.image.shape
        self.depth_current = self.depth // 2
        self.prediction = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)
        self.anotation = np.zeros((self.depth, self.height, self.width, 3), dtype=np.uint8)
        self.TL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) # height, width, depth
        self.FL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) 
        self.background_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)

        self.dice_coeff_thred = 0.75
        self.penthickness = 2

    def set_depth(self, depth):
        self.depth_current = depth

    def gray2BGRImage(self, gray_image):
        gray_image = np.uint8(cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX))
        # print(gray_image.max())
        # print(gray_image.min())
        # print(type(gray_image))
        # gray_image = (gray_image * 255).astype(np.uint8)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    def getImage2show(self):
        return cv2.addWeighted(self.gray2BGRImage(self.image[:, :, self.depth_current]), 0.9, self.anotation[self.depth_current], 0.7, 0.7)
    
    def seedsCoords2map(self):
        return self.TL_seeds[:,:,self.depth_current] * self.TL_label + self.FL_seeds[:,:,self.depth_current] * self.FL_label
        
    def prediction2anotation(self):
        for i in range(self.depth):
            self.anotation[i,:,:,:] = np.where(self.prediction[:,:,i] == self.TL_label, self.TL_color, self.anotation[i,:,:,:])
            self.anotation[i,:,:,:] = np.where(self.prediction[:,:,i] == self.FL_label, self.FL_color, self.anotation[i,:,:,:])
    
    def init_segment(self, model, device):
        self.TL_seeds[:,:,self.depth_current] = seeds2map(np.argwhere(self.anotation[self.depth_current] == self.TL_color), (self.height, self.width))
        self.FL_seeds[:,:,self.depth_current] = seeds2map(np.argwhere(self.anotation[self.depth_current] == self.FL_color), (self.height, self.width))
        window_transform_flag = True
        feature_flag = True
        sobel_flag = True

        cur_image = self.image[:,:,self.depth_current]
        last_image = self.image[:,:,self.depth_current]
        last_label = self.seedsCoords2map()
        seeds_map = last_label

        for i in range(self.depth_current, self.depth):
            cur_image = self.image[:,:,i]
            flag = True
            prediction = last_label
            if i == self.depth_current:
                indata = get_network_input_all(cur_image, np.argwhere(seeds_map > 0), seeds_map, window_transform_flag, feature_flag)
                indata = torch.from_numpy(indata).unsqueeze(0).to(device=device,dtype=torch.float32)
                prediction = get_prediction_all(model, indata)
                prediction = np.uint8(prediction)
            else:
                flag, prediction,seeds_map = get_prediction_all_bidirectional(last_label, cur_image, last_image, window_transform_flag, feature_flag, sobel_flag, self.prediction, i - self.depth_current, device, model, seeds_case = 0)
            if not flag:
                break
            # print(np.unique(prediction, return_counts = True))
            # print(prediction.shape)
            self.prediction[:,:,i] = prediction
            self.TL_seeds[:,:,i] = np.where(seeds_map == self.TL_label, self.TL_label, self.TL_seeds[:,:,i])
            self.FL_seeds[:,:,i] = np.where(seeds_map == self.FL_label, self.FL_label, self.FL_seeds[:,:,i])
            if prediction.max() < 0.5:
                break
            cur_piece = i
            cur_coeff = accuracy_all_numpy(self.prediction[:,:,cur_piece-1], self.prediction[:,:,cur_piece])
            while cur_piece > 0 and cur_coeff  < self.dice_coeff_thred:
                roll_flag, roll_prediction, roll_seeds_map = get_prediction_all_bidirectional(self.prediction[:,:,cur_piece], self.image[:,:,cur_piece-1], self.image[:,:,cur_piece], window_transform_flag, feature_flag, sobel_flag, self.prediction, 1, device, model, seeds_case = 0)
                if not roll_flag:
                    break
                if accuracy_all_numpy(self.prediction[:,:,cur_piece - 1], roll_prediction) < 0.98:
                    self.prediction[:,:,cur_piece - 1] = roll_prediction
                    self.TL_seeds[:,:,cur_piece - 1] = np.where(roll_seeds_map == self.TL_label, self.TL_label, self.TL_seeds[:,:,cur_piece - 1])
                    self.FL_seeds[:,:,cur_piece - 1] = np.where(roll_seeds_map == self.FL_label, self.FL_label, self.FL_seeds[:,:,cur_piece - 1])
                else:
                    break
                if roll_prediction.max() < 0.5:
                    break
                cur_piece = cur_piece - 1
                cur_coeff = accuracy_all_numpy(self.prediction[:,:,cur_piece-1], self.prediction[:,:,cur_piece])
            last_image = self.image[:,:,i]
            last_label = prediction
            print(f'cur piece: [{i}/{self.depth}]')
        print("finish segmentation")
        self.prediction2anotation()

        
    def Clear(self):
        self.prediction = np.zeros((self.depth, self.height, self.width), dtype=np.uint8)
        self.anotation = np.zeros((self.depth, self.height, self.width, 3), dtype=np.uint8)
        self.TL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) # height, width, depth
        self.FL_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8) 
        self.background_seeds = np.zeros((self.height, self.width, self.depth), dtype=np.uint8)

    def savePrediction(self, save_path):
        save2h5(save_path, ['image', 'prediction'], [self.image, self.prediction])



    def anotate(self, x, y):
        """
        need to do: 先得到img再求得坐标，需要搞清楚坐标之间的关系
        prediction也需要更改，和anotation output不一样，一个是保存预测结果，一个是保存渲染结果
        """
        if self.TL_flag:
            if not self.TL_seeds[y, x, self.depth_current]:
                # print("add seed")
                # self.TL_seeds.append((y, x, self.depth_current))
                cv2.rectangle(self.anotation[self.depth_current], (x - 1, y - 1), (x + 1, y + 1), self.TL_color, self.penthickness)
        if self.FL_flag:
            if not self.FL_seeds[y, x, self.depth_current]:
                # self.FL_seeds.append((y, x, self.depth_current))
                cv2.rectangle(self.anotation[self.depth_current], (x - 1, y - 1), (x + 1, y + 1), self.FL_color, self.penthickness)
        if self.background_flag:
            if not self.background_seeds[y, x, self.depth_current]:
                # self.background_seeds.append((y, x, self.depth_current))
                cv2.rectangle(self.anotation[self.depth_current], (x - 1, y - 1), (x + 1, y + 1), self.background_color, self.penthickness)
