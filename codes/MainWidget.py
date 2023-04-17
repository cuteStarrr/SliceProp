from ast import Pass
import random
import sys
import time

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from InteractImage import *
from UNet_COPY import *

import cv2

"""
need to do:
1. 补充model的path
2. Refinement函数
"""

class MainWidget(QWidget):


    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)
        
        self.__InitData() #先初始化数据，再初始化界面
        self.__InitView()
    
    def __InitData(self):
        '''
        初始化成员变量，比如一些通用设置，图片信息需要载入图片之后才生成
        '''
        self.penthickness = 2
        self.TL_color = (0, 0, 255)
        self.FL_color = (255, 0, 0) # BGR
        self.background_color = (0, 255, 0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.segment_model_path = r'/data/xuxin/ImageTBAD_processed/training_files/two_class/bothkinds_masks/transform_sobel_scribble/U_Net_transform_sobel_scribble_loss_8.pth'
        self.refinement_model_path = ""
        self.segment_model = U_Net(3, 3)
        self.segment_model.load_state_dict(torch.load(self.segment_model_path, map_location = self.device))
        self.segment_model.to(device=self.device)
        self.refinement_model = ""
        """
        初始化model,load参数,to device, eval
        """
        # self.isAdd = 1
        # self.remove_anotation_flag = 0
        # """
        # add_seed的坐标对应方式与原始图像，也就是h5文件一致
        # """
        # self.TL_seeds = [] # height, width, depth
        # self.FL_seeds = [] 
        # self.background_seed = [] # height, width, depth
        # self.depth_current = 0
        # self.crop_size = 96
        # self.expand_size = (1024, 256, 256) # depth, height, width
        
        
    def __InitView(self):
        '''
                  初始化界面
        '''
        #self.setFixedSize(700,480)
        self.setWindowTitle("GUI")
        
        #新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self) 
        #设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10) 

        """
        这里需要改进，如何显示图像
        """
        image_layout = QVBoxLayout()
        self.PaintBoard = QLabel()
        # self.PaintBoard.setPixmap(QPixmap.fromImage(
        #     self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))
        self.PaintBoard.mousePressEvent = self.mouse_press
        self.PaintBoard.mouseMoveEvent = self.mouse_move
        image_layout.addWidget(self.PaintBoard)
        self.depth_slider = QSlider(Qt.Horizontal)
        image_layout.addWidget(self.depth_slider)
        self.slider_label = QLabel()
        image_layout.addWidget(self.slider_label)
        main_layout.addLayout(image_layout)

        hbox = QVBoxLayout()
        #设置此子布局和内部控件的间距为10px
        #hbox.setContentsMargins(10, 10, 10, 10) 

        """
        和图像处理有关的layout
        """
        self.__btn_Load = QPushButton("Load Image")
        self.__btn_Load.setParent(self)
        self.__btn_Load.setStyleSheet("background-color:white")
        self.__btn_Load.clicked.connect(self.Load)

        self.__btn_Front = QPushButton("Init Segment")
        self.__btn_Front.setParent(self)
        self.__btn_Front.setStyleSheet("background-color:white")
        self.__btn_Front.clicked.connect(self.Segment)

        self.__btn_Back = QPushButton("Refinement")
        self.__btn_Back.setParent(self)
        self.__btn_Back.setStyleSheet("background-color:white")
        self.__btn_Back.clicked.connect(self.Refinement)

        self.__btn_Clear = QPushButton("Clear all")
        self.__btn_Clear.setParent(self)
        self.__btn_Clear.setStyleSheet("background-color:white")
        self.__btn_Clear.clicked.connect(self.Clear)

        self.__btn_Quit = QPushButton("Exit")
        self.__btn_Quit.setParent(self) #设置父对象为本界面
        self.__btn_Quit.setStyleSheet("background-color:white")
        self.__btn_Quit.clicked.connect(self.Quit)

        self.__btn_Save = QPushButton("Save segmentation")
        self.__btn_Save.setParent(self)
        self.__btn_Save.setStyleSheet("background-color:white")
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)

        StateLine = QLabel()
        StateLine.setText("User input.")
        palette = QPalette()
        palette.setColor(StateLine.foregroundRole(), Qt.blue)
        StateLine.setPalette(palette)

        MethodLine = QLabel()
        MethodLine.setText("Segmentation.")
        mpalette = QPalette()
        mpalette.setColor(MethodLine.foregroundRole(), Qt.blue)
        MethodLine.setPalette(mpalette)

        SaveLine = QLabel()
        SaveLine.setText("Clean or Save.")
        spalette = QPalette()
        spalette.setColor(SaveLine.foregroundRole(), Qt.blue)
        SaveLine.setPalette(spalette)

        ExitLine = QLabel()
        ExitLine.setText("Exit.")
        epalette = QPalette()
        epalette.setColor(ExitLine.foregroundRole(), Qt.blue)
        ExitLine.setPalette(epalette)

        tipsFont = StateLine.font()
        tipsFont.setPointSize(10)
        StateLine.setFixedHeight(30)
        StateLine.setWordWrap(True)
        StateLine.setFont(tipsFont)
        MethodLine.setFixedHeight(30)
        MethodLine.setWordWrap(True)
        MethodLine.setFont(tipsFont)
        SaveLine.setFixedHeight(30)
        SaveLine.setWordWrap(True)
        SaveLine.setFont(tipsFont)
        ExitLine.setFixedHeight(30)
        ExitLine.setWordWrap(True)
        ExitLine.setFont(tipsFont)

        #hbox = QVBoxLayout()
        hbox.addWidget(StateLine)
        hbox.addWidget(self.__btn_Load)
        hbox.addWidget(MethodLine)
        hbox.addWidget(self.__btn_Front)
        hbox.addWidget(self.__btn_Back)
        hbox.addWidget(SaveLine)
        hbox.addWidget(self.__btn_Clear)
        hbox.addWidget(self.__btn_Save)
        hbox.addWidget(ExitLine)
        hbox.addWidget(self.__btn_Quit)
        hbox.addStretch()
        
        #新建垂直子布局用于放置按键
        #sub_layout = QVBoxLayout() 

        splitter = QSplitter(self) #占位符
        hbox.addWidget(splitter)

        vbox = QHBoxLayout(self) 

        self.__cbtn_TL = QRadioButton("TL--1")
        self.__cbtn_TL.setParent(self)
        self.__cbtn_TL.setStyleSheet("QRadioButton{color:red}")
        self.__cbtn_TL.clicked.connect(self.on_cbtn_TL_clicked)
        vbox.addWidget(self.__cbtn_TL)
        
        self.__cbtn_FL = QRadioButton("FL--2")
        self.__cbtn_FL.setParent(self)
        self.__cbtn_FL.setStyleSheet("QRadioButton{color:blue}")
        self.__cbtn_FL.clicked.connect(self.on_cbtn_FL_clicked)
        vbox.addWidget(self.__cbtn_FL)

        self.__cbtn_Background = QRadioButton("Background")
        self.__cbtn_Background.setParent(self)
        self.__cbtn_Background.setStyleSheet("QRadioButton{color:green}")
        self.__cbtn_Background.clicked.connect(self.on_cbtn_Background_clicked)
        vbox.addWidget(self.__cbtn_Background)
        
        hbox.addLayout(vbox)
        splitter = QSplitter(self) #占位符
        hbox.addWidget(splitter)
        

        main_layout.addLayout(hbox) #将子布局加入主布局

    
    """
    保存需要改进
    """
    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Segment', '.\\', 'all files (*.*)')
        # print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        # image = self.interact_image.GetContentAsQImage()
        # image.save(savePath[0])
        self.interact_image.savePrediction(savePath[0])
        
    def on_cbtn_FL_clicked(self):
        if self.__cbtn_FL.isChecked():
            self.interact_image.FL_flag = True
            self.interact_image.TL_flag = False
            self.interact_image.background_flag = False
        else:
            self.interact_image.FL_flag = False

    def on_cbtn_TL_clicked(self):
        if self.__cbtn_TL.isChecked():
            self.interact_image.TL_flag = True
            self.interact_image.FL_flag = False
            self.interact_image.background_flag = False
        else:
            self.interact_image.TL_flag = False

    def on_cbtn_Background_clicked(self):
        if self.__cbtn_Background.isChecked():
            self.interact_image.background_flag = True #进入橡皮擦模式
            self.interact_image.FL_flag = False
            self.interact_image.TL_flag = False
        else:
            self.interact_image.background_flag = False

    def mouse_press(self, event):
        # print("mouse pressed!")
        self.interact_image.anotate(event.x(), event.y())
        # print("finish anotation!")
        self.PaintBoard.setPixmap(QPixmap.fromImage(
            self.getQImage(self.interact_image.getImage2show())))

    def mouse_move(self, event):
        # print("mouse moving!")
        self.interact_image.anotate(event.x(), event.y())
        # print("finish anotation!")
        self.PaintBoard.setPixmap(QPixmap.fromImage(
            self.getQImage(self.interact_image.getImage2show())))
        
        
    def Quit(self):
        self.close()

    # def Front(self):
    #     if self.image_data.remove_anotation_flag == 0:
    #         """
    #         每次更新都不保留之前的标注，包括add和remove
    #         """
    #         self.image_data.init_segment()
    #         self.PaintBoard.setPixmap(QPixmap.fromImage(
    #             self.getQImage(self.image_data.getImage2show())))

    def Clear(self):
        self.interact_image.Clear()
        self.PaintBoard.setPixmap(QPixmap.fromImage(
                self.getQImage(self.interact_image.getImage2show())))
            

    def Segment(self):
        self.interact_image.init_segment(self.segment_model, self.device)
        self.interact_image.prediction2anotation()
        self.PaintBoard.setPixmap(QPixmap.fromImage(
                self.getQImage(self.interact_image.getImage2show())))


    def Refinement(self):
        print("need to do")

    

    def Load(self):
        file_name = QFileDialog.getOpenFileName()
        if file_name[0] is not None and file_name[0] != "":
            # there need to be changed
            self.interact_image = InteractImage(file_name[0])
            # self.image_data.getImage2np(file_name[0])
            self.depth_slider.setMinimum(0)
            self.depth_slider.setMaximum(self.interact_image.depth - 1)
            self.depth_slider.setSingleStep(1)
            self.depth_slider.setValue(self.interact_image.depth // 2)
            # self.interact_image.set_depth(self.interact_image.depth // 2)
            self.depth_slider.setTickPosition(QSlider.TicksBelow)
            self.depth_slider.valueChanged.connect(self.depthChange)
            self.slider_label.setText("当前深度：" + str(self.interact_image.depth_current))
            self.slider_label.setFont(QFont('Arial Black', 15))
            image = QPixmap.fromImage(self.getQImage(self.interact_image.getImage2show()))
            self.PaintBoard.setPixmap(image)
            self.PaintBoard.setFixedSize(QSize(image.width(),image.height()))
            

            #self.pictureLabel.setPixmap(pic_ori)

    def getQImage(self, cvimage):
        height, width, bytes_per_pix = cvimage.shape
        bytes_per_line = width * bytes_per_pix
        cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB, cvimage)
        return QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
    def depthChange(self):
        self.interact_image.set_depth(self.depth_slider.value())
        self.slider_label.setText("当前深度：" + str(self.depth_slider.value()))
        self.PaintBoard.setPixmap(QPixmap.fromImage(self.getQImage(self.interact_image.getImage2show())))
