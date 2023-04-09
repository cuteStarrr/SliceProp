from ast import Pass
import random
import sys
import time

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ImageData import *

import cv2

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
                  初始化成员变量
        '''
        self.image_data = ImageData()
        
        
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

        self.__btn_Front = QPushButton("Frontground")
        self.__btn_Front.setParent(self)
        self.__btn_Front.setStyleSheet("background-color:white")
        self.__btn_Front.clicked.connect(self.Front)

        self.__btn_Back = QPushButton("Background")
        self.__btn_Back.setParent(self)
        self.__btn_Back.setStyleSheet("background-color:white")
        self.__btn_Back.clicked.connect(self.Back)

        self.__btn_Clear = QPushButton("Clear all")
        self.__btn_Clear.setParent(self)
        self.__btn_Clear.setStyleSheet("background-color:white")
        self.__btn_Clear.clicked.connect(self.image_data.Clear)

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
        
        self.__cbtn_Add = QRadioButton("Add")
        self.__cbtn_Add.setParent(self)
        self.__cbtn_Add.setStyleSheet("QRadioButton{color:red}")
        self.__cbtn_Add.clicked.connect(self.on_cbtn_Add_clicked)
        vbox.addWidget(self.__cbtn_Add)

        self.__cbtn_Remove = QRadioButton("Remove")
        self.__cbtn_Remove.setParent(self)
        self.__cbtn_Remove.setStyleSheet("QRadioButton{color:green}")
        self.__cbtn_Remove.clicked.connect(self.on_cbtn_Remove_clicked)
        vbox.addWidget(self.__cbtn_Remove)
        
        hbox.addLayout(vbox)
        splitter = QSplitter(self) #占位符
        hbox.addWidget(splitter)
        

        main_layout.addLayout(hbox) #将子布局加入主布局

    
    """
    保存需要改进
    """
    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Segment', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.image_data.GetContentAsQImage()
        image.save(savePath[0])
        
    def on_cbtn_Add_clicked(self):
        if self.__cbtn_Add.isChecked():
            self.image_data.isAdd = 1 
        else:
            self.image_data.isAdd = 0 

    def on_cbtn_Remove_clicked(self):
        if self.__cbtn_Remove.isChecked():
            self.image_data.isAdd = 0 #进入橡皮擦模式
        else:
            self.image_data.isAdd = 1 #退出橡皮擦模式

    def mouse_press(self, event):
        # print("mouse pressed!")
        self.image_data.anotate(event.x(), event.y(), self.image_data.isAdd)
        # print("finish anotation!")
        self.PaintBoard.setPixmap(QPixmap.fromImage(
            self.getQImage(self.image_data.getImage2show())))

    def mouse_move(self, event):
        # print("mouse moving!")
        self.image_data.anotate(event.x(), event.y(), self.image_data.isAdd)
        # print("finish anotation!")
        self.PaintBoard.setPixmap(QPixmap.fromImage(
            self.getQImage(self.image_data.getImage2show())))
        
        
    def Quit(self):
        self.close()

    def Front(self):
        if self.image_data.remove_anotation_flag == 0:
            """
            每次更新都不保留之前的标注，包括add和remove
            """
            self.image_data.init_segment()
            self.PaintBoard.setPixmap(QPixmap.fromImage(
                self.getQImage(self.image_data.getImage2show())))

    def Back(self):
        print("background")

    def Load(self):
        file_name = QFileDialog.getOpenFileName()
        if file_name[0] is not None and file_name[0] != "":
            # there need to be changed
            self.image_data.getImage2np(file_name[0])
            self.depth_slider.setMinimum(0)
            self.depth_slider.setMaximum(self.image_data.image_depth - 1)
            self.depth_slider.setSingleStep(1)
            self.depth_slider.setValue(self.image_data.image_depth // 2)
            self.image_data.depth_current = self.image_data.image_depth // 2
            self.depth_slider.setTickPosition(QSlider.TicksBelow)
            self.depth_slider.valueChanged.connect(self.depthChange)
            self.slider_label.setText("当前深度：" + str(self.image_data.depth_current))
            self.slider_label.setFont(QFont('Arial Black', 15))
            image = QPixmap.fromImage(self.getQImage(self.image_data.getImage2show()))
            self.PaintBoard.setPixmap(image)
            self.PaintBoard.setFixedSize(QSize(image.width(),image.height()))
            

            #self.pictureLabel.setPixmap(pic_ori)

    def getQImage(self, cvimage):
        height, width, bytes_per_pix = cvimage.shape
        bytes_per_line = width * bytes_per_pix
        cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB, cvimage)
        return QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
    def depthChange(self):
        self.image_data.depth_current = self.depth_slider.value()
        self.slider_label.setText("当前深度：" + str(self.depth_slider.value()))
        self.PaintBoard.setPixmap(QPixmap.fromImage(self.getQImage(self.image_data.getImage2show())))
