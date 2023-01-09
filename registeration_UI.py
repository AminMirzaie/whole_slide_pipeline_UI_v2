import os.path
from registration import *
from PyQt5.QtWidgets import QFileDialog,QMessageBox,QLabel
from loader import LoadingScreen
from multi import *
from stich import *
import concurrent.futures
import time
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.main_window = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(982, 781)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 981, 731))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 50, 971, 31))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.reg_in_text = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.reg_in_text.setAlignment(QtCore.Qt.AlignCenter)
        self.reg_in_text.setObjectName("reg_in_text")
        self.horizontalLayout_5.addWidget(self.reg_in_text)
        self.reg_in_but = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.reg_in_but.setMaximumSize(QtCore.QSize(100, 16777215))
        self.reg_in_but.setObjectName("reg_in_but")
        self.horizontalLayout_5.addWidget(self.reg_in_but)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(0, 80, 971, 31))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.reg_out_text = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.reg_out_text.setAlignment(QtCore.Qt.AlignCenter)
        self.reg_out_text.setObjectName("reg_out_text")
        self.horizontalLayout_6.addWidget(self.reg_out_text)
        self.reg_out_but = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.reg_out_but.setMaximumSize(QtCore.QSize(100, 16777215))
        self.reg_out_but.setObjectName("reg_out_but")
        self.horizontalLayout_6.addWidget(self.reg_out_but)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(-1, 119, 981, 511))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.reg_line1_label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.reg_line1_label.setMaximumSize(QtCore.QSize(100, 16777215))
        self.reg_line1_label.setAutoFillBackground(False)
        self.reg_line1_label.setStyleSheet("background:red;")
        self.reg_line1_label.setAlignment(QtCore.Qt.AlignCenter)
        self.reg_line1_label.setObjectName("reg_line1_label")
        self.horizontalLayout_11.addWidget(self.reg_line1_label)
        self.reg_line2_label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.reg_line2_label.setMaximumSize(QtCore.QSize(100, 16777215))
        self.reg_line2_label.setAutoFillBackground(False)
        self.reg_line2_label.setStyleSheet("background:green;")
        self.reg_line2_label.setAlignment(QtCore.Qt.AlignCenter)
        self.reg_line2_label.setObjectName("reg_line2_label")
        self.horizontalLayout_11.addWidget(self.reg_line2_label)
        self.reg_line3_label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.reg_line3_label.setMaximumSize(QtCore.QSize(100, 16777215))
        self.reg_line3_label.setAutoFillBackground(False)
        self.reg_line3_label.setStyleSheet("background:blue;")
        self.reg_line3_label.setAlignment(QtCore.Qt.AlignCenter)
        self.reg_line3_label.setObjectName("reg_line3_label")
        self.horizontalLayout_11.addWidget(self.reg_line3_label)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_7 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_7.setObjectName("label_7")
        self.verticalLayout_4.addWidget(self.label_7)
        self.reg_log = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.reg_log.setAutoFillBackground(False)
        self.reg_log.setStyleSheet("background:#e5f9ff;")
        self.reg_log.setText("")
        self.reg_log.setObjectName("reg_log")
        self.verticalLayout_4.addWidget(self.reg_log)
        self.verticalLayout_3.addLayout(self.verticalLayout_4)
        self.horizontalLayout_11.addLayout(self.verticalLayout_3)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_11)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(0, 20, 971, 31))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.reg_back_text = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.reg_back_text.setAlignment(QtCore.Qt.AlignCenter)
        self.reg_back_text.setObjectName("reg_back_text")
        self.horizontalLayout_8.addWidget(self.reg_back_text)
        self.reg_back_but = QtWidgets.QPushButton(self.verticalLayoutWidget_3)
        self.reg_back_but.setMaximumSize(QtCore.QSize(100, 16777215))
        self.reg_back_but.setObjectName("reg_back_but")
        self.horizontalLayout_8.addWidget(self.reg_back_but)
        self.verticalLayout_6.addLayout(self.horizontalLayout_8)
        self.reg_exec_but = QtWidgets.QPushButton(self.tab)
        self.reg_exec_but.setGeometry(QtCore.QRect(320, 650, 351, 41))
        self.reg_exec_but.setObjectName("reg_exec_but")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.tabWidget.addTab(self.tab_5, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 982, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # connect listeners................................
        self.reg_log.setAlignment(QtCore.Qt.AlignLeft)
        self.reg_in_but.setDisabled(True)
        self.reg_out_but.setDisabled(True)
        self.reg_exec_but.setDisabled(True)
        self.reg_back_but.clicked.connect(self.reg_back_but_func)
        self.reg_in_but.clicked.connect(self.reg_in_but_func)
        self.reg_out_but.clicked.connect(self.reg_out_but_func)
        self.reg_exec_but.clicked.connect(self.reg_exec_but_func)
        self.loader = LoadingScreen()

    def ex_listener(self, f):
        out_dir = f.result()
        self.main_window.setDisabled(False)
        self.reg_log.setText(self.reg_log.text() + "transformation matrix saved in " + out_dir+" ! \n")
        self.reg_log.setText(self.reg_log.text() + "process done!\n")
        self.loader.stopAnimation()

    def coef_listener(self,data):
        self.reg_coefs = data
        self.loader.stopAnimation()
        self.reg_log.setText(self.reg_log.text() + "background images loaded successfuly!\n")
        self.reg_back_text.setText(self.coef_path)
        self.main_window.setDisabled(False)
        self.reg_in_but.setDisabled(False)

    def template_listener(self,data):
        self.R,self.G,self.B = data
        self.main_window.setDisabled(False)
        self.reg_log.setText(self.reg_log.text() + "template images loaded successfuly!\n")
        self.reg_in_text.setText(self.template_line)
        self.reg_out_but.setDisabled(False)
        self.reg_in_text.setText(self.template_line)
        self.loader.stopAnimation()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.reg_in_text.setText(_translate("MainWindow", "specify the first line of template file:"))
        self.reg_in_but.setText(_translate("MainWindow", "browse"))
        self.reg_out_text.setText(_translate("MainWindow", "specify the output directory to save registration file:"))
        self.reg_out_but.setText(_translate("MainWindow", "browse"))
        self.reg_line1_label.setText(_translate("MainWindow", "line 1"))
        self.reg_line2_label.setText(_translate("MainWindow", "line 2"))
        self.reg_line3_label.setText(_translate("MainWindow", "line 3"))
        self.label_7.setText(_translate("MainWindow", "log:"))
        self.reg_back_text.setText(_translate("MainWindow", "specify the first line of background images:"))
        self.reg_back_but.setText(_translate("MainWindow", "browse"))
        self.reg_exec_but.setText(_translate("MainWindow", "execute"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "registeration"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "calibration"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "stich"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "tile maker"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "upload"))

    def reg_back_but_func(self):
        self.openFileNamesDialog_back()

    def reg_in_but_func(self):
        self.openFileNamesDialog_in()

    def reg_out_but_func(self):
        self.openFileNamesDialog_out()

    def message(self,text):
        msg = QMessageBox()
        msg.setWindowTitle("error")
        msg.setText(text)
        x = msg.exec_()

    def openFileNamesDialog_back(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.coef_path, _ = QFileDialog.getOpenFileName(self.reg_in_but, "QFileDialog.getOpenFileName()", "",
                                                                "All Files (*);;Python Files (*.py)", options=options)
        if not(self.coef_path.split(".")[-1] == "tif"):
           self.message("you should select tif file!")
        else:
            try:
                self.loader.startAnimation()
                self.reg_log.setText(self.reg_log.text() + "laoding the background images...\n")
                self.thread_coef = QtCore.QThread()
                self.worker_coef = coef_worker(self.coef_path)
                self.worker_coef.moveToThread(self.thread_coef)
                self.thread_coef.started.connect(self.worker_coef.run)
                self.worker_coef.finished.connect(self.thread_coef.quit)
                self.worker_coef.finished.connect(self.worker_coef.deleteLater)
                self.thread_coef.finished.connect(self.thread_coef.deleteLater)
                self.worker_coef.finished2.connect(self.coef_listener)
                self.thread_coef.start()
                self.main_window.setDisabled(True)

            except:
                self.message("background images is not image file!")
                self.reg_log.setText(self.reg_log.text() + "error in reading the background images!\n")

    def openFileNamesDialog_in(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.template_line, _ = QFileDialog.getOpenFileName(self.reg_in_but, "QFileDialog.getOpenFileName()", "",
                                                    "All Files (*);;Python Files (*.py)", options=options)
        if not(self.template_line.split(".")[-1] == "tif"):
           self.message("you should select tif file!")
        else:
            self.reg_log.setText(self.reg_log.text() + "laoding the template images...\n")
            try:
                self.thread_temp = QtCore.QThread()
                self.worker_temp = template_worker(self.template_line,self.reg_coefs,self)
                self.worker_temp.moveToThread(self.thread_temp)
                self.thread_temp.started.connect(self.worker_temp.run)
                self.worker_temp.finished.connect(self.thread_temp.quit)
                self.worker_temp.finished.connect(self.worker_temp.deleteLater)
                self.thread_temp.finished.connect(self.thread_temp.deleteLater)
                self.worker_temp.finished2.connect(self.template_listener)
                self.thread_temp.start()
                self.loader.startAnimation()
                self.reg_log.setText(self.reg_log.text() + "template images writed in temp directory!\n")
                self.main_window.setDisabled(True)
            except:
                self.message("background images is not image file!")
                self.reg_log.setText(self.reg_log.text() + "error in reading the background images!\n")

    def openFileNamesDialog_out(self):
        try:
            homography_out_dir = str(QFileDialog.getExistingDirectory(self.reg_out_but, "Select Directory"))+"/"
            if os.path.exists(homography_out_dir + "transformation_matrix"):
                self.homography_out_dir = homography_out_dir + "transformation_matrix/"
            else:
                os.mkdir(homography_out_dir + "transformation_matrix")
                self.homography_out_dir = homography_out_dir + "transformation_matrix/"
            self.reg_log.setText(self.reg_log.text() + "output directory specified correctly!\n")
            self.reg_out_text.setText(self.homography_out_dir)
            self.reg_exec_but.setDisabled(False)
        except:
            self.message("you should select output directory to save transformation!")

    def reg_exec_but_func(self):
        # executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        # future = executor.submit(process_homographies, self.R, self.G, self.B, self.homography_out_dir)
        # future.add_done_callback(self.ex_listener)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        scher = Sticher("/home/humangene/Desktop/test2/", "/home/humangene/share/place2/blood404/","/home/humangene/share/place2/BG_cB.tif",10)
        future = executor.submit(scher.stich_lines)
        future.add_done_callback(self.amin)
        self.reg_log.setText(self.reg_log.text() + "finding the transformation matrix!\n")
        self.reg_log.setText(self.reg_log.text() + "this will take some time!\n")
        self.loader.startAnimation()
        self.main_window.setDisabled(True)
    def amin(self,f):
        data = f.result()
        print(data)
        self.main_window.setDisabled(False)
        self.loader.stopAnimation()
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
