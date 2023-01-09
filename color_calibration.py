import os

import numpy as np
import time
from stich import *
import cv2
from PyQt5 import QtCore, QtGui
import tifffile
class calibrator_worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    finished2 = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(str)
    def __init__(self, BR_h, GR_h, real_image_directory,
              wsi_directory, background_directory, card_directory,
              thread_number, color_names, logger,
              real_card_viewer, before_card_viewer, after_card_viewer):
        super().__init__()
        self.BR_h = BR_h
        self.GR_h = GR_h
        self.real_image_directory = real_image_directory
        self.wsi_directory = wsi_directory
        self.background_directory = background_directory
        if not os.path.exists(card_directory+"cards"):
            os.mkdir(card_directory + "cards")
        self.card_directory = card_directory+"cards/"
        self.thread_number = thread_number
        self.color_names = color_names
        self.logger = logger
        self.real_card_viewer = real_card_viewer
        self.before_card_viewer = before_card_viewer
        self.after_card_viewer = after_card_viewer

    def run(self):
        self.progress.emit("processing real colors!\n")
        n = len(self.color_names)
        tile_size = 256
        space = 100
        row_size = n * tile_size + (n + 1) * space
        col_size = tile_size + space
        real_card = (np.ones((row_size, col_size, 3)) * 255).astype("uint8")
        wsi_card = (np.ones((row_size, col_size, 3)) * 255).astype("uint8")
        mask = (np.zeros((row_size, col_size))).astype("uint8")
        rcolor_means = {}
        for color in self.color_names:
            self.progress.emit("\t procrssing image "+color+" !\n")
            tot = tifffile.imread(self.real_image_directory + color + ".tif")
            middle = tot.shape[0] // 2, tot.shape[1] // 2
            rcolor_means[color] = np.mean(tot[middle[0] - tot.shape[0] // 3:middle[0] + tot.shape[0] // 3,
                                          middle[1] - tot.shape[1] // 3:middle[1] + tot.shape[1] // 3], axis=(0, 1))
        count = 0
        for color in self.color_names:
            start_row = ((count + 1) * space) + (count * tile_size)
            start_col = 50
            real_card[start_row:start_row + tile_size, start_col:start_col + tile_size, :] = rcolor_means[color]
            mask[start_row:start_row + tile_size, start_col:start_col + tile_size] = count + 1
            count += 1
        tifffile.imwrite(self.card_directory+"real_color_card.png", real_card)
        tifffile.imwrite(self.card_directory + "mask.png", mask)
        r_image = QtGui.QImage(self.card_directory+"real_color_card.png")
        r_image = r_image.scaled(self.real_card_viewer.width(), self.real_card_viewer.height(),
                                 aspectRatioMode=QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                 transformMode=QtCore.Qt.SmoothTransformation)
        self.real_card_viewer.setPixmap(QtGui.QPixmap.fromImage(r_image))
        self.progress.emit("processing wsi colors!\n")
        color_means = {}
        if not(os.path.exists(self.card_directory+"wsi_images")):
            os.mkdir(self.card_directory+"wsi_images")
        for color in self.color_names:
            print(color)
            base_folder = self.wsi_directory + color + "/"
            if not (color == "bg"):
                self.progress.emit("\tprocessing the lines of color " + color + " !\n")
                sticer = calib_Sticher(self.BR_h, self.GR_h, base_folder, self.background_directory, self.thread_number)
                tot = sticer.stich_lines()

                tifffile.imwrite(self.card_directory+"wsi_images/" + color + ".tif", tot)
                middle = tot.shape[0] // 2, tot.shape[1] // 2
                color_means[color] = np.mean(tot[middle[0] - tot.shape[0] // 3:middle[0] + tot.shape[0] // 3,
                                             middle[1] - tot.shape[1] // 3:middle[1] + tot.shape[1] // 3], axis=(0, 1))
            else:
                self.progress.emit("\tprocessing the lines of color " + color + " !\n")
                images = {}
                BR_h_arr = np.loadtxt(self.BR_h)
                GR_h_arr = np.loadtxt(self.GR_h)
                self.coefs = generateCoefficents(self.background_directory)
                R = cv2.imread(base_folder + "BG_cR.tif", cv2.IMREAD_UNCHANGED)[:, :-10]
                G = cv2.imread(base_folder + "BG_cG.tif", cv2.IMREAD_UNCHANGED)[:, :-10]
                B = cv2.imread(base_folder + "BG_cB.tif", cv2.IMREAD_UNCHANGED)[:, :-10]
                R_e = rescale(R * self.coefs[0])
                G_e = align_channel(rescale(G * self.coefs[1]), GR_h_arr)
                B_e = align_channel(rescale(B * self.coefs[2]), BR_h_arr)
                images[0] = np.clip(
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
                tot = images[0]
                middle = tot.shape[0] // 2, tot.shape[1] // 2
                tifffile.imwrite(self.card_directory+"wsi_images/" + color + ".tif", tot)
                color_means[color] = np.mean(tot[middle[0] - tot.shape[0] // 3:middle[0] + tot.shape[0] // 3,
                                             middle[1] - tot.shape[1] // 3:middle[1] + tot.shape[1] // 3], axis=(0, 1))

        count = 0
        for color in self.color_names:
            start_row = ((count + 1) * space) + (count * tile_size)
            start_col = 50
            wsi_card[start_row:start_row + tile_size, start_col:start_col + tile_size, :] = color_means[color]
            count += 1
        tifffile.imwrite(self.card_directory+"wsi_color_card.png", wsi_card)
        w_image = QtGui.QImage(self.card_directory+"wsi_color_card.png")
        w_image = w_image.scaled(self.before_card_viewer.width(), self.before_card_viewer.height(),
                                 aspectRatioMode=QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                 transformMode=QtCore.Qt.SmoothTransformation)
        self.before_card_viewer.setPixmap(QtGui.QPixmap.fromImage(w_image))


        self.progress.emit("finding the color transformation matrix !\n")
        real_card = tifffile.imread(self.card_directory+"real_color_card.png")
        wsi_card = tifffile.imread(self.card_directory+"wsi_color_card.png")
        mask = tifffile.imread(self.card_directory + "mask.png")

        target_headers, target_matrix = self.get_color_matrix(real_card, mask)
        source_headers, source_matrix = self.get_color_matrix(wsi_card, mask)
        matrix_a, matrix_m, matrix_b = self.get_matrix_m(target_matrix=target_matrix, source_matrix=source_matrix)
        transformation_matrix = self.calc_transformation_matrix(matrix_m, matrix_b)
        self.progress.emit("apply color transformation matrix on wsi card !\n")
        corrected = self.apply_transformation_matrix(wsi_card, transformation_matrix)
        tifffile.imwrite(self.card_directory + "corrected_card.png", corrected)

        c_image = QtGui.QImage(self.card_directory+"corrected_card.png")
        c_image = c_image.scaled(self.after_card_viewer.width(), self.after_card_viewer.height(),
                                 aspectRatioMode=QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                 transformMode=QtCore.Qt.SmoothTransformation)
        self.after_card_viewer.setPixmap(QtGui.QPixmap.fromImage(c_image))
        np.savetxt(self.card_directory+"color_calib_transformation.txt", transformation_matrix)
        self.progress.emit("color transformation_matrix saved in "+self.card_directory+ "color_calib_transformation.txt !\n")
        self.progress.emit("done!")
        self.finished.emit()
        self.finished2.emit()

    def get_color_matrix(self,rgb_img, mask):
        img_dtype = rgb_img.dtype
        max_val = 1.0
        if img_dtype.kind == 'u':
            max_val = np.iinfo(img_dtype).max
        rgb_img = rgb_img.astype(np.float64) / max_val
        color_matrix = np.zeros((len(np.unique(mask)) - 1, 4))
        headers = ["chip_number", "r_avg", "g_avg", "b_avg"]
        row_counter = 0
        for i in np.unique(mask):
            if i != 0:
                chip = rgb_img[np.where(mask == i)]
                color_matrix[row_counter][0] = i
                color_matrix[row_counter][1] = np.mean(chip[:, 2])
                color_matrix[row_counter][2] = np.mean(chip[:, 1])
                color_matrix[row_counter][3] = np.mean(chip[:, 0])
                row_counter += 1
        return headers, color_matrix

    def get_matrix_m(self, target_matrix, source_matrix):
        t_cc, t_r, t_g, t_b = np.split(target_matrix, 4, 1)
        s_cc, s_r, s_g, s_b = np.split(source_matrix, 4, 1)
        ones = np.ones_like(s_b)
        matrix_a = np.concatenate((s_r, s_g, s_b, ones), 1)
        matrix_m = np.linalg.solve(np.matmul(matrix_a.T, matrix_a), matrix_a.T)
        matrix_b = np.concatenate((t_r, t_g, t_b), 1)
        return matrix_a, matrix_m, matrix_b

    def calc_transformation_matrix(self, matrix_m, matrix_b):
        transformation_matrix = np.matmul(matrix_m, matrix_b)
        return transformation_matrix

    def apply_transformation_matrix(self, source_img, transformation_matrix):
        red, green, blue = np.split(transformation_matrix, 3, 1)
        source_dtype = source_img.dtype
        max_val = 1.0
        if source_dtype.kind == 'u':
            max_val = np.iinfo(source_dtype).max
        source_flt = source_img.astype(np.float64) / max_val
        source_b, source_g, source_r = cv2.split(source_flt)
        b = blue[3] + source_r * blue[0] + source_g * blue[1] + source_b * blue[2]
        g = green[3] + source_r * green[0] + source_g * green[1] + source_b * green[2]
        r = red[3] + source_r * red[0] + source_g * red[1] + source_b * red[2]
        bgr = [b, g, r]
        corrected_img = cv2.merge(bgr)
        corrected_img = max_val * np.clip(corrected_img, 0, 1)
        corrected_img = corrected_img.astype(source_dtype)
        return corrected_img
