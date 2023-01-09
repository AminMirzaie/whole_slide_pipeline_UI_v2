import time

import cv2
import numpy as np
import os
from pystackreg import StackReg
import skimage
from skimage.transform import warp


def align_channel(chanel, m):
    a = skimage.transform.SimilarityTransform(matrix=m)
    return np.ma.array(warp(chanel, a, preserve_range=True, output_shape=chanel.shape, cval=-1))

def generateCoefficents(back_dir):
    base_folder = "_".join(back_dir.split("_")[:-1])
    R = (cv2.imread(base_folder+"_cR.tif",cv2.IMREAD_UNCHANGED)[:,:-10])
    G = (cv2.imread(base_folder+"_cG.tif",cv2.IMREAD_UNCHANGED)[:,:-10])
    B = (cv2.imread(base_folder+"_cB.tif",cv2.IMREAD_UNCHANGED)[:,:-10])
    coeficients = []
    ref_mean = 40000
    coeficients.append((1/np.mean(R,axis=0))*ref_mean)
    coeficients.append((1/np.mean(G,axis=0))*ref_mean)
    coeficients.append((1/np.mean(B,axis=0))*ref_mean)
    return coeficients


def clipper(channel):
    a = np.quantile(channel,0.01)
    b = np.quantile(channel,0.99)
    return np.clip(channel,a,b)


def rescale(channel):
    return channel*(255.0/pow(2,16))

def get_gradient(im) :
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def homography_calculator2(template,moving):
    warp_mode = cv2.MOTION_HOMOGRAPHY
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3000,  1e-3)
    (cc, warp_matrix) = cv2.findTransformECC (get_gradient(template), get_gradient(moving),warp_matrix, warp_mode, criteria)
    print (warp_matrix)
    return warp_matrix


def get_template_reg(template_dir, coeficients):
    base_folder = "_".join(template_dir.split("_")[:-1])
    R =np.clip(rescale((cv2.imread(base_folder+"_c1.tif",cv2.IMREAD_UNCHANGED)[:,:-10])*coeficients[0]),0,255).astype("uint8")
    G =np.clip(rescale((cv2.imread(base_folder+"_c2.tif",cv2.IMREAD_UNCHANGED)[:,:-10])*coeficients[1]),0,255).astype("uint8")
    B =np.clip(rescale((cv2.imread(base_folder+"_c3.tif",cv2.IMREAD_UNCHANGED)[:,:-10])*coeficients[2]),0,255).astype("uint8")
    if os.path.exists("./temp"):
        pass
    else:
        os.mkdir("./temp")
    cv2.imwrite("./temp/reg_R.png", R)
    cv2.imwrite("./temp/reg_G.png", G)
    cv2.imwrite("./temp/reg_B.png", B)
    return R, G, B

def process_homographies(r, g, b, out_dir):
    if r.shape[0] > 32000:
        middle = r.shape[0]//2
        r = r[middle-16000 : middle + 16000]
        g = g[middle - 16000: middle + 16000]
        b = b[middle - 16000: middle + 16000]
    BR_h = homography_calculator2(r, b)
    GR_h = homography_calculator2(r, g)
    np.savetxt(out_dir+"BR_h.txt", BR_h)
    np.savetxt(out_dir+"GR_h.txt", GR_h)
    return out_dir