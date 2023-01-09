import sys
from PyQt5.QtWidgets import QApplication,QWidget,QLabel
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import QMovie
from PyQt5 import QtCore
class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(200,200)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)
        self.label_animation = QLabel(self)
        self.movie = QMovie("./resources/loader.gif")
        self.movie.setScaledSize(QtCore.QSize(200,200))
        self.label_animation.setMovie(self.movie)
    def startAnimation(self):
        self.movie.start()
        self.show()
    def stopAnimation(self):
        self.movie.stop()
        self.close()
