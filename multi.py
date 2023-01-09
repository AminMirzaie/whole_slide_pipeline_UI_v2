import concurrent.futures
from registration import *
import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets

import concurrent.futures
import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets


class coef_worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    finished2 = QtCore.pyqtSignal(list)
    def __init__(self, coef_path):
        super().__init__()
        self.coef_path = coef_path

    def run(self):
        self.coefs = generateCoefficents(self.coef_path)
        self.finished2.emit(self.coefs)
        self.finished.emit()

class template_worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    finished2 = QtCore.pyqtSignal(list)
    def __init__(self, template_line,coefs,obj):
        super().__init__()
        self.coefs = coefs
        self.template_line = template_line
        self.obj = obj

    def run(self):
        R,G,B = get_template_reg(self.template_line, self.coefs)

        R_image = QtGui.QImage("./temp/reg_R.png")
        R_image = R_image.scaled(self.obj.reg_line1_label.width(), self.obj.reg_line1_label.height(),
                                 aspectRatioMode=QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                 transformMode=QtCore.Qt.SmoothTransformation)
        self.obj.reg_line1_label.setPixmap(QtGui.QPixmap.fromImage(R_image))

        G_image = QtGui.QImage("./temp/reg_G.png")
        G_image = G_image.scaled(self.obj.reg_line2_label.width(), self.obj.reg_line2_label.height(),
                                 aspectRatioMode=QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                 transformMode=QtCore.Qt.SmoothTransformation)
        self.obj.reg_line2_label.setPixmap(QtGui.QPixmap.fromImage(G_image))

        B_image = QtGui.QImage("./temp/reg_B.png")
        B_image = B_image.scaled(self.obj.reg_line3_label.width(), self.obj.reg_line3_label.height(),
                                 aspectRatioMode=QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                 transformMode=QtCore.Qt.SmoothTransformation)
        self.obj.reg_line3_label.setPixmap(QtGui.QPixmap.fromImage(B_image))

        self.finished2.emit([R,G,B])
        self.finished.emit()


class ex_worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    finished2 = QtCore.pyqtSignal(str)
    def __init__(self, R, G, B, homography_out_dir):
        super().__init__()
        self.R = R
        self.G = G
        self.B = B
        self.homography_out_dir = homography_out_dir
    def run(self):
        out_dir = process_homographies(self.R, self.G, self.B, self.homography_out_dir)
        self.finished2.emit(out_dir)
        self.finished.emit()