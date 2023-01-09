from registration import generateCoefficents, clipper, rescale, align_channel
import glob
import numpy as np
import cv2
import gc
import time
import concurrent.futures
import image_registration
from registration import align_channel,generateCoefficents,clipper,rescale
import os

import numpy as np
import time
from stich import *
import cv2
from PyQt5 import QtCore, QtGui
import tifffile

class calib_Sticher:
    def __init__(self, BR_h_path, GR_h_path, base_folder, background_path, thread_number, out_dir_path = None):
        self.BR_h = np.loadtxt(BR_h_path)
        self.GR_h = np.loadtxt(GR_h_path)
        self.base_folder = base_folder
        self.num_lines = len(glob.glob(self.base_folder + "*.tif")) // 3
        self.images = {}
        self.line_shifts = {}
        self.coefs = generateCoefficents(background_path)
        self.thread_number = thread_number

    def reading_images(self,img_num):
        print(img_num)
        R = clipper(cv2.imread(self.base_folder + "line_" + str(img_num) + "_c1.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
        G = clipper(cv2.imread(self.base_folder + "line_" + str(img_num) + "_c2.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
        B = clipper(cv2.imread(self.base_folder + "line_" + str(img_num) + "_c3.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
        R_e = rescale(R * self.coefs[0])
        G_e = align_channel(rescale(G * self.coefs[1]), self.GR_h)
        B_e = align_channel(rescale(B * self.coefs[2]), self.BR_h)
        self.images[img_num] = np.clip(
            cv2.merge([
                (R_e / np.mean(R_e, axis=0)) * np.mean(R_e),
                (G_e / np.mean(G_e, axis=0)) * np.mean(G_e),
                (B_e / np.mean(B_e, axis=0)) * np.mean(B_e)
            ]), 0, 255).astype("uint8")[5:-5, 5:-10]

        del (R)
        del (G)
        del (B)
        del (R_e)
        del (G_e)
        del (B_e)
        gc.collect()
        return img_num

    def process_image(self,img_num):
        print(img_num)
        right_image = self.images[img_num]
        left_image = self.images[img_num + 1]
        min_row = min(right_image.shape[0], left_image.shape[0])
        min_col = min(right_image.shape[1], left_image.shape[1])
        right_image_c = right_image[:min_row, :300]
        left_image_c = left_image[:min_row, -300:]
        im1 = cv2.cvtColor(left_image_c, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(right_image_c, cv2.COLOR_BGR2GRAY)
        yoff, xoff, yer, xer = image_registration.chi2_shift(im1, im2, 0.1)
        print(yoff, xoff)
        if yoff > 30 and xoff < 0:
            self.line_shifts[(img_num, img_num + 1)] = (int(yoff), int(-1 * xoff))

    def stich_lines(self):
        img_nums = list(range(self.num_lines))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_number) as executor:
            results = executor.map(self.reading_images, img_nums)
        concurrent.futures.as_completed(results)

        img_nums = list(range(self.num_lines - 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_number) as executor:
            results = list(executor.map(self.process_image, img_nums))
        concurrent.futures.as_completed(results)
        if len(self.line_shifts.keys()) > 0:
            ysum = 0
            xsum = 0
            for key in self.line_shifts.keys():
                ysum += self.line_shifts[key][0]
                xsum += self.line_shifts[key][1]
            y_m = ysum // len(self.line_shifts.keys())
            x_m = xsum // len(self.line_shifts.keys())
            for i in range(self.num_lines - 1):
                if not ((i, i + 1) in self.line_shifts.keys()):
                    self.line_shifts[(i, i + 1)] = (y_m, x_m)
        else:
            for i in range(self.num_lines - 1):
                self.line_shifts[(i, i + 1)] = (0, 0)
        sum_x = 0
        sum_y = 0
        for key, value in self.line_shifts.items():
            sum_x += value[1]
            sum_y += value[0]

        valid_list = []
        for key, value in self.line_shifts.items():
            valid_list.append(key[1])
        yoff_sum = 0
        xoff_sum = 0
        tempelate = self.images[0]
        max_tot_row = tempelate.shape[0]
        max_tot_col = tempelate.shape[1]
        for val in self.line_shifts.values():
            yoff_sum += val[0]
            xoff_sum += val[1]
        img_shape = (max_tot_row, max_tot_col, 3)
        mid = (int(img_shape[0] / 2), int(img_shape[1] / 2))
        row_numbers = img_shape[0]
        tot = np.zeros((row_numbers + xoff_sum, img_shape[1] * self.num_lines - yoff_sum, 3), dtype=np.uint8)
        last_row_index = tot.shape[0]
        last_col_index = tot.shape[1]
        valid_range = [last_row_index, last_row_index]
        for i in range(self.num_lines):
            indx = self.num_lines - i - 1
            print(indx)
            if i in valid_list:
                middle_img = self.images[i]
                img_shape = (middle_img.shape[0], middle_img.shape[1], 3)
                row_numbers = middle_img.shape[0]
                yshift, xshift = self.line_shifts[(i - 1, i)]
                tot[last_row_index - row_numbers - xshift:last_row_index - xshift,
                last_col_index - img_shape[1] + yshift:last_col_index + yshift, :] = middle_img
                valid_range[0] = last_row_index - row_numbers
                valid_range[1] = last_row_index + xshift
                last_row_index = last_row_index - xshift
                last_col_index = last_col_index - img_shape[1] + yshift
            else:
                middle_img = self.images[i]
                tot[last_row_index - middle_img.shape[0]:last_row_index,
                last_col_index - middle_img.shape[1]:last_col_index, :] = middle_img
                valid_range[0] = last_row_index - middle_img.shape[0]
                valid_range[1] = last_row_index

                last_row_index = last_row_index
                last_col_index = last_col_index - middle_img.shape[1]
        return tot


class Sticher:
    def __init__(self, BR_h_path, GR_h_path, base_folder, background_path, thread_number, whiten_back, color_calib_exist, color_calib, progress, out_dir_path = None):
        self.BR_h = np.loadtxt(BR_h_path)
        self.GR_h = np.loadtxt(GR_h_path)
        self.progress = progress
        self.base_folder = base_folder
        self.num_lines = len(glob.glob(self.base_folder + "*.tif")) // 3
        self.images = {}
        self.line_shifts = {}
        self.coefs = self.generateCoefficents(background_path)
        back_base_folder = "_".join(background_path.split("_")[:-1])
        self.thread_number = thread_number
        self.whiten_back = whiten_back
        self.color_calib_exist = color_calib_exist
        self.color_calib = color_calib
        self.out_dir_path = out_dir_path
        self.R_w = None
        self.G_w = None
        self.B_w = None
        self.R_c_w = None
        self.G_c_w = None
        self.B_c_w = None
        self.ccm = None
        if self.color_calib_exist and self.whiten_back:
            ccm = self.color_calib
            ccm_coef = ccm[:3]
            ccm_b = ccm[3]
            self.ccm = np.fliplr(np.vstack((np.flipud(ccm_coef), ccm_b)))
            Rb = (cv2.imread(back_base_folder + "_cR.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
            Gb = (cv2.imread(back_base_folder + "_cG.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
            Bb = (cv2.imread(back_base_folder + "_cB.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
            self.R_w = pow(2, 16) / np.mean(Rb * self.coefs[0])
            self.G_w = pow(2, 16) / np.mean(Gb * self.coefs[1])
            self.B_w = pow(2, 16) / np.mean(Bb * self.coefs[2])
            back = np.clip(
                cv2.merge([
                    np.clip(rescale(Rb * self.coefs[0]), 0, 255),
                    align_channel(np.clip(rescale(Gb * self.coefs[1]), 0, 255), self.GR_h),
                    align_channel(np.clip(rescale(Bb * self.coefs[2]), 0, 255), self.BR_h)
                ]), 0, 255).astype("uint8")[5:-5, 5:-10]
            s = back.shape
            back = (back.reshape((s[0] * s[1], 3)).astype(np.float64)) / 255
            back = np.concatenate((back, np.ones((back.shape[0])).reshape(-1, 1)), axis=1)
            back = np.matmul(back, ccm).reshape((s[0], s[1], 3))
            back = (255 * np.clip(back, 0, 1)).astype("uint8")
            Rb = back[:, :, 0]
            Gb = back[:, :, 1]
            Bb = back[:, :, 2]
            self.R_c_w = 255 / np.mean(Rb)
            self.G_c_w = 255 / np.mean(Gb)
            self.B_c_w = 255 / np.mean(Bb)
        elif self.color_calib_exist and not(self.whiten_back):
            ccm = self.color_calib
            ccm_coef = ccm[:3]
            ccm_b = ccm[3]
            self.ccm = np.fliplr(np.vstack((np.flipud(ccm_coef), ccm_b)))
        elif not (self.color_calib_exist) and self.whiten_back:
            Rb = (cv2.imread(back_base_folder + "_cR.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
            Gb = (cv2.imread(back_base_folder + "_cG.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
            Bb = (cv2.imread(back_base_folder + "_cB.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
            self.R_w = pow(2, 16) / np.mean(Rb * self.coefs[0])
            self.G_w = pow(2, 16) / np.mean(Gb * self.coefs[1])
            self.B_w = pow(2, 16) / np.mean(Bb * self.coefs[2])
        else:
            print("00")

    def generateCoefficents(self,back_dir):
        base_folder = "_".join(back_dir.split("_")[:-1])
        R = (cv2.imread(base_folder + "_cR.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
        G = (cv2.imread(base_folder + "_cG.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
        B = (cv2.imread(base_folder + "_cB.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
        coeficients = []
        ref_mean = 40000
        coeficients.append((1 / np.mean(R, axis=0)) * ref_mean)
        coeficients.append((1 / np.mean(G, axis=0)) * ref_mean)
        coeficients.append((1 / np.mean(B, axis=0)) * ref_mean)
        return  coeficients

    def reading_images(self,img_num):
        print(img_num)
        self.progress.emit("read and processing line number : "+str(img_num)+"!\n")
        R = clipper(cv2.imread(self.base_folder + "line_" + str(img_num) + "_c1.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
        G = clipper(cv2.imread(self.base_folder + "line_" + str(img_num) + "_c2.tif", cv2.IMREAD_UNCHANGED)[:, :-10])
        B = clipper(cv2.imread(self.base_folder + "line_" + str(img_num) + "_c3.tif", cv2.IMREAD_UNCHANGED)[:, :-10])

        if self.color_calib_exist and self.whiten_back:
            self.images[img_num] = np.clip(
                cv2.merge([
                    rescale(R * self.coefs[0]),
                    align_channel(rescale(G * self.coefs[1]), self.GR_h),
                    align_channel(rescale(B * self.coefs[2]), self.BR_h)
                ]), 0, 255).astype("uint8")[5:-5, 5:-10]
            s = self.images[img_num].shape
            self.images[img_num] = (self.images[img_num].reshape((s[0] * s[1], 3)).astype(np.float64)) / 255
            self.images[img_num] = np.concatenate((self.images[img_num], np.ones((self.images[img_num].shape[0])).reshape(-1, 1)),
                                             axis=1)
            self.images[img_num] = np.matmul(self.images[img_num], self.ccm).reshape((s[0], s[1], 3))
            self.images[img_num] = (255 * np.clip(self.images[img_num], 0, 1)).astype("uint8")
            self.images[img_num] = cv2.merge(
                [np.clip(self.images[img_num][:, :, 0] * self.R_c_w, 0, 255), np.clip(self.images[img_num][:, :, 1] * self.G_c_w, 0, 255),
                 np.clip(self.images[img_num][:, :, 2] * self.B_c_w, 0, 255)]).astype("uint8")
        elif self.color_calib_exist  and not (self.whiten_back):
            self.images[img_num] = np.clip(
                cv2.merge([
                    np.clip(rescale(R * self.coefs[0]), 0, 255),
                    align_channel(np.clip(rescale(G * self.coefs[1]), 0, 255), self.GR_h),
                    align_channel(np.clip(rescale(B * self.coefs[2]), 0, 255), self.BR_h)
                ]), 0, 255).astype("uint8")[5:-5, 5:-10]
            s = self.images[img_num].shape
            self.images[img_num] = (self.images[img_num].reshape((s[0] * s[1], 3)).astype(np.float64)) / 255
            self.images[img_num] = np.concatenate((self.images[img_num], np.ones((self.images[img_num].shape[0])).reshape(-1, 1)),
                                             axis=1)
            self.images[img_num] = np.matmul(self.images[img_num], self.ccm).reshape((s[0], s[1], 3))
            self.images[img_num] = (255 * np.clip(self.images[img_num], 0, 1)).astype("uint8")
        elif not (self.color_calib_exist) and self.whiten_back:
            self.images[img_num] = np.clip(
                cv2.merge([
                    np.clip(rescale(R * self.coefs[0] * self.R_w), 0, 255),
                    align_channel(np.clip(rescale(G * self.coefs[1] * self.G_w), 0, 255), self.GR_h),
                    align_channel(np.clip(rescale(B * self.coefs[2] * self.B_w), 0, 255), self.BR_h)
                ]), 0, 255).astype("uint8")[5:-5, 5:-10]
        else:
            print("00")
            self.images[img_num] = np.clip(
                cv2.merge([
                    np.clip(rescale(R * self.coefs[0]), 0, 255),
                    align_channel(np.clip(rescale(G * self.coefs[1]), 0, 255), self.GR_h),
                    align_channel(np.clip(rescale(B * self.coefs[2]), 0, 255), self.BR_h)
                ]), 0, 255).astype("uint8")[5:-5, 5:-10]
        self.progress.emit("read and processing line number : " + str(img_num) + " done!\n")
    def process_image(self,img_num):
        print(img_num)
        self.progress.emit("stiching line: " + str(img_num) + " and line: "+str(img_num+1)+"!\n")
        right_image = self.images[img_num]
        left_image = self.images[img_num + 1]
        min_row = min(right_image.shape[0], left_image.shape[0])
        min_col = min(right_image.shape[1], left_image.shape[1])
        right_image_c = right_image[:min_row, :300]
        left_image_c = left_image[:min_row, -300:]
        im1 = cv2.cvtColor(left_image_c, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(right_image_c, cv2.COLOR_BGR2GRAY)
        yoff, xoff, yer, xer = image_registration.chi2_shift(im1, im2, 0.1)
        print(yoff, xoff)
        if yoff > 30 and yoff < 50 and xoff < 0 and xoff > -30:
            self.line_shifts[(img_num, img_num + 1)] = (int(yoff), int(-1 * xoff))
        self.progress.emit("stiching line: " + str(img_num) + " and line: " + str(img_num + 1) + " done !\n")
    def stich_lines(self):
        self.progress.emit("creating the stiched image!\n")
        img_nums = list(range(self.num_lines))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_number) as executor:
            results = executor.map(self.reading_images, img_nums)
        concurrent.futures.as_completed(results)

        img_nums = list(range(self.num_lines - 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_number) as executor:
            results = list(executor.map(self.process_image, img_nums))
        concurrent.futures.as_completed(results)
        if len(self.line_shifts.keys()) > 0:
            ysum = 0
            xsum = 0
            for key in self.line_shifts.keys():
                ysum += self.line_shifts[key][0]
                xsum += self.line_shifts[key][1]
            y_m = ysum // len(self.line_shifts.keys())
            x_m = xsum // len(self.line_shifts.keys())
            for i in range(self.num_lines - 1):
                if not ((i, i + 1) in self.line_shifts.keys()):
                    self.line_shifts[(i, i + 1)] = (y_m, x_m)
        else:
            for i in range(self.num_lines - 1):
                self.line_shifts[(i, i + 1)] = (0, 0)
        sum_x = 0
        sum_y = 0
        for key, value in self.line_shifts.items():
            sum_x += value[1]
            sum_y += value[0]

        valid_list = []
        for key, value in self.line_shifts.items():
            valid_list.append(key[1])
        yoff_sum = 0
        xoff_sum = 0
        tempelate = self.images[0]
        max_tot_row = tempelate.shape[0]
        max_tot_col = tempelate.shape[1]
        for val in self.line_shifts.values():
            yoff_sum += val[0]
            xoff_sum += val[1]
        img_shape = (max_tot_row, max_tot_col, 3)
        mid = (int(img_shape[0] / 2), int(img_shape[1] / 2))
        row_numbers = img_shape[0]
        tot = np.zeros((row_numbers + xoff_sum, img_shape[1] * self.num_lines - yoff_sum, 3), dtype=np.uint8)
        last_row_index = tot.shape[0]
        last_col_index = tot.shape[1]
        valid_range = [last_row_index, last_row_index]
        for i in range(self.num_lines):
            indx = self.num_lines - i - 1
            print(indx)
            if i in valid_list:
                middle_img = self.images[i]
                img_shape = (middle_img.shape[0], middle_img.shape[1], 3)
                row_numbers = middle_img.shape[0]
                yshift, xshift = self.line_shifts[(i - 1, i)]
                tot[last_row_index - row_numbers - xshift:last_row_index - xshift,
                last_col_index - img_shape[1] + yshift:last_col_index + yshift, :] = middle_img
                valid_range[0] = last_row_index - row_numbers
                valid_range[1] = last_row_index + xshift
                last_row_index = last_row_index - xshift
                last_col_index = last_col_index - img_shape[1] + yshift
            else:
                middle_img = self.images[i]
                tot[last_row_index - middle_img.shape[0]:last_row_index,
                last_col_index - middle_img.shape[1]:last_col_index, :] = middle_img
                valid_range[0] = last_row_index - middle_img.shape[0]
                valid_range[1] = last_row_index

                last_row_index = last_row_index
                last_col_index = last_col_index - middle_img.shape[1]
        self.progress.emit("stiched image created!\n")
        return tot

class stich_worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    finished2 = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(str)
    def __init__(self, BR_h, GR_h, base_folder, background_directory, out_directory,
              thread_number, logger, image_viewer, whiten_back, color_calib_exist, color_calib_mat):
        super().__init__()
        self.BR_h = BR_h
        self.GR_h = GR_h
        self.base_folder = base_folder
        self.background_directory = background_directory
        self.out_directory = out_directory
        if not os.path.exists(self.out_directory+"stiched"):
            os.mkdir(self.out_directory + "stiched")
        self.out_directory = self.out_directory+"stiched/"
        self.thread_number = thread_number
        self.logger = logger
        self.image_viewer = image_viewer
        self.whiten_back = whiten_back
        self.color_calib_exist = color_calib_exist
        self.color_calib_mat = color_calib_mat

    def run(self):
        self.progress.emit("processing lines of image!\n")
        sticer = Sticher(self.BR_h, self.GR_h, self.base_folder, self.background_directory, self.thread_number, self.whiten_back, self.color_calib_exist, self.color_calib_mat,self.progress)
        tot = sticer.stich_lines()
        self.progress.emit("stiching image done!\n")
        self.progress.emit("saving image in "+self.out_directory+" ...\n")
        image_name = self.base_folder.split("/")[-2]+".tif"
        print(self.base_folder)
        print(image_name)
        tifffile.imwrite(self.out_directory+"/"+image_name, tot)
        self.progress.emit("image saved in " + self.out_directory +image_name+"!\n")
        w_image = QtGui.QImage(self.out_directory +image_name)
        w_image = w_image.scaled(self.image_viewer.width(), self.image_viewer.height(),
                                 aspectRatioMode=QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                 transformMode=QtCore.Qt.SmoothTransformation)
        self.image_viewer.setPixmap(QtGui.QPixmap.fromImage(w_image))
        self.finished.emit()
        self.finished2.emit()
