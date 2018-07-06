# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from GraphCut import GraphMaker
from PIL import Image
from DEXTR import utils
import torch
from collections import OrderedDict
from DEXTR import deeplab_resnet as resnet
from torch.nn.functional import upsample
from torch.autograd import Variable
from mypath import Path

class CutUI(QWidget):

    def __init__(self, image_dir=None, result_dir=None, mode='dextr'):
        super().__init__()

        self.image_dir = image_dir
        self.result_dir = result_dir

        path = self.result_dir

        if not os.path.exists(path):
            os.mkdir(path)

        file_list = os.listdir(self.image_dir)

        self.images = file_list
        self.image_name = self.images[0]
        self.id = 0
        self.lenths = len(file_list) - 1
        self.mode = mode
        self.gpu_id = -1
        self.pad = 50
        self.thres = 0.8
        self.modelName = 'dextr_pascal-sbd'

        self.image = cv2.imread(os.path.join(self.image_dir,self.image_name))
        self.overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.temp_overlay = np.zeros_like(self.image)
        self.extreme_points_ori = []
        self.dextr_results = []
        self.completed_image_list = []

        self.seed_type = 1
        self.seedStartX = 0
        self.seedStartY = 0
        self.seedReleaseX = 0
        self.seedReleaseY = 0
        self.IsEraser = False
        self.IsAdd = False
        self.flag = True

        self.__InitUI()

        if self.mode == 'graphcut':
            print('Using GRAPH CUT')
            self.graphcutButton.setStyleSheet("background-color:gray")
            self.dextrButton.setStyleSheet("background-color:white")

        elif self.mode == 'dextr':
            print('Using DEXTR CUT')
            self.graphcutButton.setStyleSheet("background-color:white")
            self.dextrButton.setStyleSheet("background-color:gray")

    def __InitUI(self):
        self.graphcutButton = QPushButton("Graphcut")
        self.graphcutButton.setStyleSheet("background-color:white")
        self.graphcutButton.clicked.connect(self.on_graphcut)

        self.dextrButton = QPushButton("DEXTR")
        self.dextrButton.setStyleSheet("background-color:white")
        self.dextrButton.clicked.connect(self.on_dextr)

        segmentButton = QPushButton("Segmentation")
        segmentButton.setStyleSheet("background-color:white")
        segmentButton.clicked.connect(self.on_segment)

        finishButton = QPushButton("Finish")
        finishButton.setStyleSheet("background-color:white")
        finishButton.clicked.connect(self.on_finish)

        nextButton = QPushButton("Next")
        nextButton.setStyleSheet("background-color:white")
        nextButton.clicked.connect(self.on_next)

        lastButton = QPushButton("Last")
        lastButton.setStyleSheet("background-color:white")
        lastButton.clicked.connect(self.on_last)

        self.thinkness = QLineEdit("3")
        self.thinkness.setStyleSheet("background-color:white")
        self.thinkness.setMaximumWidth(30)

        clearButton = QPushButton("Clear All")
        clearButton.setStyleSheet("background-color:white")
        clearButton.clicked.connect(self.on_clear)

        self.eraserButton = QPushButton("Eraser")
        self.eraserButton.setStyleSheet("background-color:white")
        self.eraserButton.clicked.connect(self.on_eraser)

        self.addButton = QPushButton("Add")
        self.addButton.setStyleSheet("background-color:white")
        self.addButton.clicked.connect(self.on_add)

        hbox = QHBoxLayout()

        hbox.addWidget(self.dextrButton)
        hbox.addWidget(self.graphcutButton)
        hbox.addWidget(segmentButton)
        hbox.addWidget(self.addButton)
        hbox.addWidget(self.eraserButton)
        hbox.addWidget(clearButton)
        hbox.addWidget(finishButton)
        hbox.addWidget(lastButton)
        hbox.addWidget(nextButton)
        hbox.addWidget(self.thinkness)

        hbox.addStretch(1)

        self.seedLabel = QLabel()


        self.seedLabel.mousePressEvent = self.mouse_down
        self.seedLabel.mouseReleaseEvent = self.mouse_release
        self.seedLabel.mouseMoveEvent = self.mouse_drag

        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.get_image_with_overlay())))


        imagebox = QHBoxLayout()
        imagebox.addWidget(self.seedLabel)

        vbox = QVBoxLayout()

        vbox.addLayout(hbox)
        vbox.addLayout(imagebox)
        vbox.addStretch()

        self.setLayout(vbox)

        self.setWindowTitle('Segmentation')
        self.show()

    @staticmethod
    def get_qimage(cvimage):
        height, width, bytes_per_pix = cvimage.shape
        bytes_per_line = width * bytes_per_pix

        cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB, cvimage)
        return QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def mouse_down(self, event):
        thinkness = int(self.thinkness.text())

        if event.button() == Qt.LeftButton:
            self.seed_type = 1
        elif event.button() == Qt.RightButton:
            self.seed_type = 0

        temp_overlay = self.get_image_with_overlay()
        self.seedStartX = event.x()
        self.seedStartY = event.y()
        self.seedReleaseX = event.x()
        self.seedReleaseY = event.y()

        if self.mode == 'graphcut':

            if not self.IsAdd and not self.IsEraser:

                if self.seed_type == 1:
                    cv2.circle(temp_overlay, (self.seedStartX, self.seedStartY), int(thinkness / 2), (255, 255, 255),
                               int(thinkness / 2))  #
                else:
                    cv2.circle(temp_overlay, (self.seedStartX, self.seedStartY), int(thinkness / 2), (0, 0, 255),
                               int(thinkness / 2))

                self.seedLabel.setPixmap(QPixmap.fromImage(
                    self.get_qimage(temp_overlay)))

            elif self.IsEraser:
                if len(np.unique(self.segment_overlay)) <= 1:
                    print('You cannot wipe any pixels until you finish at least one segmentation!')
                else:
                    cv2.circle(self.segment_overlay, (self.seedStartX, self.seedStartY), int(thinkness / 2), (0, 0, 0),
                               int(thinkness / 2))

                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

            elif self.IsAdd:
                if len(np.unique(self.segment_overlay)) <= 1:
                    print('You cannot add any pixels until you finish at least one segmentation!')
                else:
                    cv2.circle(self.segment_overlay, (self.seedStartX, self.seedStartY), int(thinkness / 2),
                               (255, 255, 255),
                               int(thinkness / 2))
                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

        elif self.mode == 'dextr':

            if not self.IsAdd and not self.IsEraser:
                if len(np.unique(self.segment_overlay)) <= 1:

                    if len(self.extreme_points_ori) < 4:
                        cv2.circle(self.overlay, (self.seedStartX, self.seedStartY), int(thinkness / 2), (0, 255, 0),
                                   4)
                        self.seedLabel.setPixmap(QPixmap.fromImage(
                            self.get_qimage(self.get_image_with_overlay())))
                        self.extreme_points_ori.append((self.seedStartX, self.seedStartY))

                    else:
                        self.seedLabel.setPixmap(QPixmap.fromImage(
                            self.get_qimage(self.get_image_with_overlay())))
                        print('You can only input 4 extreme points!')

                else:
                    if len(self.extreme_points_ori) < 4:
                        cv2.circle(self.segment_overlay, (self.seedStartX, self.seedStartY), int(thinkness / 2),
                                   (0, 255, 0),
                                   4)
                        self.seedLabel.setPixmap(QPixmap.fromImage(
                            self.get_qimage(self.get_image_with_overlay(1))))
                        self.extreme_points_ori.append((self.seedStartX, self.seedStartY))

                    else:
                        self.seedLabel.setPixmap(QPixmap.fromImage(
                            self.get_qimage(self.get_image_with_overlay(1))))
                        print('You can only input 4 extreme points!')

            elif self.IsEraser:
                if len(np.unique(self.segment_overlay)) <= 1:
                    print('You cannot wipe any pixels until you finish at least one segmentation!')
                else:
                    cv2.circle(self.segment_overlay, (self.seedStartX, self.seedStartY), int(thinkness / 2), (0, 0, 0),
                               int(thinkness / 2))

                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

            elif self.IsAdd:
                if len(np.unique(self.segment_overlay)) <= 1:
                    print('You cannot add any pixels until you finish at least one segmentation!')
                else:
                    cv2.circle(self.segment_overlay, (self.seedStartX, self.seedStartY), int(thinkness / 2),
                               (255, 255, 255),
                               int(thinkness / 2))
                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

    def mouse_drag(self, event):
        thinkness = int(self.thinkness.text())

        self.seedReleaseX = event.x()
        self.seedReleaseY = event.y()
        temp_overlay = self.get_image_with_overlay()

        if self.mode == 'graphcut':
            if not self.IsAdd and not self.IsEraser:
                if self.seed_type == 1:
                    cv2.line(temp_overlay, (self.seedStartX, self.seedStartY), (self.seedReleaseX, self.seedReleaseY),
                             (255, 255, 255), thinkness)
                else:
                    cv2.line(temp_overlay, (self.seedStartX, self.seedStartY), (self.seedReleaseX, self.seedReleaseY),
                             (0, 0, 255), thinkness)

                self.seedLabel.setPixmap(QPixmap.fromImage(
                    self.get_qimage(temp_overlay)))

            elif self.IsEraser:
                if len(np.unique(self.segment_overlay)) <= 1:
                    pass
                else:
                    cv2.line(self.segment_overlay, (self.seedStartX, self.seedStartY),
                             (self.seedReleaseX, self.seedReleaseY),
                             (0, 0, 0), thinkness)

                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

            elif self.IsAdd:
                if len(np.unique(self.segment_overlay)) <= 1:
                    pass
                else:
                    cv2.line(self.segment_overlay, (self.seedStartX, self.seedStartY),
                             (self.seedReleaseX, self.seedReleaseY),
                             (255, 255, 255), thinkness)

                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

        elif self.mode == 'dextr':
            if not self.IsAdd and not self.IsEraser:
                pass

            elif self.IsEraser:
                if len(np.unique(self.segment_overlay)) <= 1:
                    pass
                else:
                    cv2.line(self.segment_overlay, (self.seedStartX, self.seedStartY),
                             (self.seedReleaseX, self.seedReleaseY),
                             (0, 0, 0), thinkness)

                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

            elif self.IsAdd:
                if len(np.unique(self.segment_overlay)) <= 1:
                    pass
                else:
                    cv2.line(self.segment_overlay, (self.seedStartX, self.seedStartY),
                             (self.seedReleaseX, self.seedReleaseY),
                             (255, 255, 255), thinkness)

                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

    def mouse_release(self, event):
        thinkness = int(self.thinkness.text())

        if self.mode == 'graphcut':
            if not self.IsAdd and not self.IsEraser:

                if self.seed_type == 1:
                    cv2.line(self.overlay, (self.seedStartX, self.seedStartY), (self.seedReleaseX, self.seedReleaseY),
                             (255, 255, 255), thinkness)
                else:
                    cv2.line(self.overlay, (self.seedStartX, self.seedStartY), (self.seedReleaseX, self.seedReleaseY),
                             (0, 0, 255), thinkness)

                self.seedLabel.setPixmap(QPixmap.fromImage(
                    self.get_qimage(self.get_image_with_overlay())))

            elif self.IsEraser:
                if len(np.unique(self.segment_overlay)) <= 1:
                    pass
                else:
                    cv2.line(self.segment_overlay, (self.seedStartX, self.seedStartY),
                             (self.seedReleaseX, self.seedReleaseY),
                             (0, 0, 0), thinkness)
                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))


            elif self.IsAdd:
                if len(np.unique(self.segment_overlay)) <= 1:
                    pass
                else:
                    cv2.line(self.segment_overlay, (self.seedStartX, self.seedStartY),
                             (self.seedReleaseX, self.seedReleaseY),
                             (255, 255, 255), thinkness)
                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

        elif self.mode == 'dextr':
            if not self.IsAdd and not self.IsEraser:
                pass

            elif self.IsEraser:
                if len(np.unique(self.segment_overlay)) <= 1:
                    pass
                else:
                    cv2.line(self.segment_overlay, (self.seedStartX, self.seedStartY),
                             (self.seedReleaseX, self.seedReleaseY),
                             (0, 0, 0), thinkness)
                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

            elif self.IsAdd:
                if len(np.unique(self.segment_overlay)) <= 1:
                    pass
                else:
                    cv2.line(self.segment_overlay, (self.seedStartX, self.seedStartY),
                             (self.seedReleaseX, self.seedReleaseY),
                             (255, 255, 255), thinkness)
                    self.seedLabel.setPixmap(QPixmap.fromImage(
                        self.get_qimage(self.get_image_with_overlay(1))))

    @pyqtSlot()
    def on_graphcut(self):
        if self.mode != 'graphcut':
            print('Using GRAPH CUT')
            self.mode = 'graphcut'
        self.IsAdd = False
        self.IsEraser = False
        self.eraserButton.setStyleSheet("background-color:white")
        self.addButton.setStyleSheet("background-color:white")
        self.segment_overlay = np.zeros_like(self.image)
        self.extreme_points_ori = []
        self.overlay = np.zeros_like(self.image)
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.get_image_with_overlay())))
        self.graphcutButton.setStyleSheet("background-color:gray")
        self.dextrButton.setStyleSheet("background-color:white")

    @pyqtSlot()
    def on_dextr(self):
        if self.mode != 'dextr':
            print('Using DEXTR CUT')
            self.mode = 'dextr'
        self.IsAdd = False
        self.IsEraser = False
        self.eraserButton.setStyleSheet("background-color:white")
        self.addButton.setStyleSheet("background-color:white")
        self.segment_overlay = np.zeros_like(self.image)
        self.extreme_points_ori = []
        self.overlay = np.zeros_like(self.image)
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.get_image_with_overlay())))
        self.graphcutButton.setStyleSheet("background-color:white")
        self.dextrButton.setStyleSheet("background-color:gray")

    @pyqtSlot()
    def on_eraser(self):
        if not self.IsEraser:
            self.IsEraser = True
            self.IsAdd = False
            self.addButton.setStyleSheet("background-color:white")
            self.eraserButton.setStyleSheet("background-color:gray")
        else:
            self.IsEraser = False
            self.eraserButton.setStyleSheet("background-color:white")

    @pyqtSlot()
    def on_add(self):
        if not self.IsAdd:
            self.IsAdd = True
            self.IsEraser = False
            self.eraserButton.setStyleSheet("background-color:white")
            self.addButton.setStyleSheet("background-color:gray")
        else:
            self.IsAdd = False
            self.addButton.setStyleSheet("background-color:white")

    @pyqtSlot()
    def on_segment(self):
        if self.mode == 'graphcut':
            graph_maker = GraphMaker.GraphMaker(self.image)
            height, width = np.shape(self.overlay)[:2]
            for i in range(height):
                for j in range(width):
                    if self.overlay[i, j, 0] != 0 or self.overlay[i, j, 1] != 0 or self.overlay[i, j, 2] != 0:
                        if self.overlay[i, j, 0] == 0 and self.overlay[i, j, 2] >= 200:
                            graph_maker.add_seed(j, i, 0)
                        elif self.overlay[i, j, 1] >= 200:
                            graph_maker.add_seed(j, i, 1)
            graph_maker.create_graph()
            self.segment_overlay = graph_maker.segment_overlay

            self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.get_image_with_overlay(1))))


        elif self.mode == 'dextr':
            if self.flag:
                if torch.cuda.is_available():
                    self.gpu_id = 0

                self.net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
                print("Initializing weights from: {}".format(os.path.join(Path.models_dir(), self.modelName + '.pth')))
                state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), self.modelName + '.pth'),
                                                   map_location=lambda storage, loc: storage)
                # Remove the prefix .module from the model when it is trained using DataParallel
                if 'module.' in list(state_dict_checkpoint.keys())[0]:
                    new_state_dict = OrderedDict()
                    for k, v in state_dict_checkpoint.items():
                        name = k[7:]  # remove `module.` from multi-gpu training
                        new_state_dict[name] = v
                else:
                    new_state_dict = state_dict_checkpoint
                self.net.load_state_dict(new_state_dict)
                self.net.eval()
                if self.gpu_id >= 0:
                    torch.cuda.set_device(device=self.gpu_id)
                    self.net.cuda()
                self.flag = False

            image = np.array(self.image)
            height, width = np.shape(image)[:2]
            b, g, r = cv2.split(image)
            image = cv2.merge([r, g, b])
            extreme_points_ori = np.array(self.extreme_points_ori).astype(np.int)
            bbox = utils.get_bbox(image, points=extreme_points_ori, pad=self.pad, zero_pad=True)
            crop_image = utils.crop_from_bbox(image, bbox, zero_pad=True)
            resize_image = utils.fixed_resize(crop_image, (512, 512)).astype(np.float32)

            #  Generate extreme point heat map normalized to image values
            extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]),
                                                   np.min(extreme_points_ori[:, 1])] + [self.pad,
                                                                                        self.pad]
            extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
            extreme_heatmap = utils.make_gt(resize_image, extreme_points, sigma=10)
            extreme_heatmap = utils.cstm_normalize(extreme_heatmap, 255)

            #  Concatenate inputs and convert to tensor
            input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
            input_dextr = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

            # Run a forward pass
            inputs = Variable(input_dextr, requires_grad=True)
            if self.gpu_id >= 0:
                inputs = inputs.cuda()

            with torch.no_grad():
                outputs = self.net.forward(inputs)
            outputs = upsample(outputs, size=(height, width), mode='bilinear', align_corners=True)
            if self.gpu_id >= 0:
                outputs = outputs.cpu()

            pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            result = utils.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True,
                                           relax=self.pad) > self.thres

            self.dextr_results.append(result)
            self.extreme_points_ori = []

            # self.segment_overlay = np.zeros((np.shape(image)))

            overall_result = np.zeros_like(result)
            for seg_result in self.dextr_results:
                overall_result += seg_result
            self.segment_overlay[:, :, 0][overall_result == True] = 255
            self.segment_overlay[:, :, 0][overall_result == False] = 0

            self.segment_overlay[:, :, 1][overall_result == True] = 255
            self.segment_overlay[:, :, 1][overall_result == False] = 0

            self.segment_overlay[:, :, 2][overall_result == True] = 255
            self.segment_overlay[:, :, 2][overall_result == False] = 0

            self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.get_image_with_overlay(1))))

    @pyqtSlot()
    def on_next(self):
        self.IsAdd = False
        self.IsEraser = False
        self.eraserButton.setStyleSheet("background-color:white")
        self.addButton.setStyleSheet("background-color:white")
        self.dextr_results = []
        self.extreme_points_ori = []

        self.id += 1
        if self.id > self.lenths:
            self.id = 0
        self.image_name = self.images[self.id]
        self.image = cv2.imread(os.path.join(self.image_dir,self.image_name))
        self.segment_overlay = np.zeros_like(self.image)
        self.overlay = np.zeros_like(self.image)

        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.get_image_with_overlay())))

    @pyqtSlot()
    def on_last(self):
        self.IsAdd = False
        self.IsEraser = False
        self.eraserButton.setStyleSheet("background-color:white")
        self.addButton.setStyleSheet("background-color:white")
        self.dextr_results = []
        self.extreme_points_ori = []

        self.id -= 1
        if self.id < 0:
            self.id = self.lenths
        self.image_name = self.images[self.id]
        self.image = cv2.imread(os.path.join(self.image_dir,self.image_name))
        self.segment_overlay = np.zeros_like(self.image)
        self.overlay = np.zeros_like(self.image)

        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.get_image_with_overlay())))

    @pyqtSlot()
    def on_clear(self):
        self.IsAdd = False
        self.IsEraser = False
        self.eraserButton.setStyleSheet("background-color:white")
        self.addButton.setStyleSheet("background-color:white")
        self.segment_overlay = np.zeros_like(self.image)
        self.extreme_points_ori = []
        self.dextr_results = []
        self.overlay = np.zeros_like(self.image)
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.get_image_with_overlay())))

    @pyqtSlot()
    def on_finish(self):
        # TODO: finish segmentation, save result and turn to next image

        save_path = os.path.join(self.result_dir, self.image_name[:-4]+'.png')
        cv2.imwrite(save_path, self.segment_overlay)

        self.dextr_results = []

        self.IsAdd = False
        self.IsEraser = False
        self.eraserButton.setStyleSheet("background-color:white")
        self.addButton.setStyleSheet("background-color:white")

        if self.lenths > 1:
            print("{} images remained.".format(str(self.lenths)))
        else:
            print("{} image remained.".format(str(self.lenths)))

        if self.lenths <= 0:
            print("Segmentation completed. Please close the window!")

        else:
            self.lenths -= 1
            self.completed_image_list.append(self.images[self.id])
            del self.images[self.id]
            self.id += 1
            if self.id > self.lenths:
                self.id = 0
            self.image_name = self.images[self.id]
            self.image = cv2.imread(os.path.join(self.image_dir, self.image_name))
            self.segment_overlay = np.zeros_like(self.image)
            self.overlay = np.zeros_like(self.image)

            self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.get_image_with_overlay())))

    def get_image_with_overlay(self, show_mode=0):
        if show_mode == 0:
            return cv2.addWeighted(self.image, 0.7, self.overlay, 1, 0.1)
        elif show_mode == 1:
            return cv2.addWeighted(self.image, 0.9, self.segment_overlay.astype(np.uint8), 0.6, 0.1)
        else:
            print('wrong number!')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CutUI(Path.db_root_dir(), Path.save_root_dir())
    app.exec_()

