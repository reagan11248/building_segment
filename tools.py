##########################
# 常用函数存放在此脚本内
#
##########################
# -*- coding: utf-8 -*-
import warnings
# warnings.filterwarnings("ignore")

from osgeo import gdal, ogr, osr
import time
import math
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import os
import sys

os.environ['PROJ_LIB'] = os.path.dirname(sys.argv[0])

import re
import shutil
import cv2
from skimage import morphology, filters, draw
from skimage.feature import local_binary_pattern, canny, graph
from skimage.segmentation import felzenszwalb, slic, mark_boundaries, watershed
from skimage.io import imread, imshow
from skimage.segmentation import slic, quickshift, felzenszwalb
from skimage.measure import regionprops, label
from skimage.filters import gabor
from skimage.color import rgb2gray

from sklearn import metrics, preprocessing, svm, model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, MiniBatchSparsePCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier


import scipy
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import multiprocessing
import multiprocessing.sharedctypes
from networkx.linalg import adjacency_matrix

'''
0    GDT_Unknown : 未知数据类型
1    GDT_Byte : 8bit正整型 (C++中对应unsigned char)
2    GDT_UInt16 : 16bit正整型 (C++中对应 unsigned short)
3    GDT_Int16 : 16bit整型 (C++中对应 short 或 short int)
4    GDT_UInt32 : 32bit 正整型 (C++中对应unsigned long)
5    GDT_Int32 : 32bit整型 (C++中对应int 或 long 或 long int)
6    GDT_Float32 : 32bit 浮点型 (C++中对应float)
7    GDT_Float64 : 64bit 浮点型 (C++中对应double)
8    GDT_CInt16 : 16bit复整型 (?)
9    GDT_CInt32 : 32bit复整型 (?)
10   GDT_CFloat32 : 32bit复浮点型 (?)
11   GDT_CFloat64 : 64bit复浮点型 (?)
'''



# 打开tif
def open_tif(path):
    dataset = gdal.Open(path)
    if dataset is None:
        print('erro:', path)
    cols = dataset.RasterXSize  # 图像长度
    rows = (dataset.RasterYSize)  # 图像宽度
    geotransform = dataset.GetGeoTransform()
    space = dataset.GetProjection()
    return dataset, cols, rows, geotransform, space


# 创建tif
def creat_tif(path, array, space, geotransform, datatype=1):
    driver = gdal.GetDriverByName("GTiff")
    shape = np.shape(array)

    if len(shape) == 2:
        s = [1, shape[0], shape[1]]
    else:
        s = shape

    outdata = driver.Create(path, s[2], s[1], s[0], datatype)
    if outdata == None:
        print('error:', path)
    if s[0] == 1:
        outdata.GetRasterBand(1).WriteArray(array)
    else:
        for i in range(s[0]):
            outdata.GetRasterBand(i + 1).WriteArray(array[i])
    outdata.SetProjection(space)  # 投影信息
    outdata.SetGeoTransform(geotransform)
    outdata.FlushCache()


# 分块
def clip(img_file, out_path, X, Y, overlap, datatype=1):
    dataset, cols, rows, geotransform, space = open_tif(img_file)

    c = list(range(0, cols, math.ceil(cols / X)))
    r = list(range(0, rows, math.ceil(rows / Y)))
    c.append(cols)
    r.append(rows)

    for i in range(X):
        for j in range(Y):
            x1 = c[i]
            y1 = r[j]
            if i + 1 == X:
                x2 = c[i + 1]
            else:
                x2 = c[i + 1] + overlap
            if j + 1 == Y:
                y2 = r[j + 1]
            else:
                y2 = r[j + 1] + overlap

            path = out_path + '/' + str(i) + '_' + str(j) + '.tif'
            array1 = dataset.GetRasterBand(1).ReadAsArray(x1, y1, x2 - x1, y2 - y1)

            array2 = dataset.GetRasterBand(2).ReadAsArray(x1, y1, x2 - x1, y2 - y1)
            array3 = dataset.GetRasterBand(3).ReadAsArray(x1, y1, x2 - x1, y2 - y1)
            array = np.array([array1, array2, array3])
            geotransform1 = (geotransform[0] + geotransform[1] * x1,
                             geotransform[1],
                             geotransform[2],
                             geotransform[3] + geotransform[5] * y1,
                             geotransform[4],
                             geotransform[5])

            creat_tif(path, array, space, geotransform1, datatype)

# slic
def slic_segment(avg):
    img_path, ndvi_path, out_path = avg
    print('start:', os.path.basename(img_path))
    dataset, cols, rows, geotransform, space = open_tif(img_path)
    img = dataset.ReadAsArray()
    img = img.transpose(1, 2, 0)
    #img = cv2.bilateralFilter(src=img, d=7, sigmaColor=40, sigmaSpace=40)

    dataset, cols, rows, geotransform, space = open_tif(ndvi_path)
    ndvi = dataset.ReadAsArray()
    mask = np.zeros_like(ndvi)
    mask[ndvi < 0.3] = 255
    mask = mask.astype(np.uint8)
    mask = morphology.remove_small_holes(mask, 500)
    mask = morphology.opening(mask, morphology.disk(3))


    n_segments = (np.shape(img)[0] // 30) * (np.shape(img)[1] // 30)
    labels = slic(img, n_segments=n_segments, compactness=20, max_iter=30)

    #g = graph.rag_mean_color(img, labels)
    #labels = graph.cut_threshold(labels, g, 10)
   # print('segment_over:', os.path.basename(img_path))
    '''
    slic_img = cv2.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler=20.0)
    slic_img.iterate(10)
    labels = slicslic_img.getLabels()
    '''
    id = np.unique(labels.flatten())
    new_id = [i for i in range(len(id))]
    new_label = np.zeros_like(labels)
    d = dict(zip(id, new_id))
    co, ro = np.shape(labels)
    # print('label:', co, ro)
    for i in range(co):
        for j in range(ro):
            new_label[i, j] = d[labels[i, j]]
   
    creat_tif(out_path, new_label, space, geotransform, 4)

    print('process_over:', os.path.basename(img_path))


def slic_segment_muti(path):
    if not os.path.exists(path + '/temporary/label'):
        #shutil.rmtree(path + '/temporary/label')
        os.mkdir(path + '/temporary/label')

    tif_list = os.listdir(path + '/org/img')
    '''
    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            
            img_path = path + '/temporary/index/' + os.path.splitext(filename)[0] + '/color.tif'
            ndvi_path = path + '/temporary/index/' + os.path.splitext(filename)[0] + '/ndvi.tif'
            out_path = path + '/temporary/label/' + filename
            arg = [img_path, ndvi_path, out_path]
            slic_segment(arg)
    '''
    p = multiprocessing.Pool(4)
    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            img_path = path + '/temporary/index/' + os.path.splitext(filename)[0] + '/color.tif'
            out_path = path + '/temporary/label/' + filename
            arg = [img_path, out_path]
            p.apply_async(slic_segment, args=(arg, ))
    p.close()
    p.join()


# 区域统计
def block_mea_std(index_img, segment_img, out_path):
    index_dataset, cols, rows, geotransform, space = open_tif(index_img)
    index_array = index_dataset.GetRasterBand(1).ReadAsArray()
    segment_dataset, cols, rows, geotransform, space = open_tif(segment_img)
    segment_array = segment_dataset.GetRasterBand(1).ReadAsArray()

    l = np.unique(segment_array)

    h_arr = np.zeros((len(l), 2), dtype=np.float32)

    for idx, val in enumerate(l):
        l_v = index_array[segment_array == val]
        h_arr[idx] = np.array([np.mean(l_v), np.var(l_v)])

    filename = os.path.splitext(os.path.basename(index_img))[0]
    print(np.shape(h_arr)[0])
    np.savez(out_path + '/' + filename + '_mean_std.npz', h_arr)


def block_histogram(index_img, segment_img, histogram_x, out_path):  # histogram_x:(unique)
    index_dataset, cols, rows, geotransform, space = open_tif(index_img)
    index_array = index_dataset.GetRasterBand(1).ReadAsArray()
    segment_dataset, cols, rows, geotransform, space = open_tif(segment_img)
    segment_array = segment_dataset.GetRasterBand(1).ReadAsArray()

    l = np.unique(segment_array)
    h_arr = np.zeros((len(l), len(histogram_x)), dtype=np.float32)

    for idx, val in enumerate(l):
        u_c = np.unique(index_array[segment_array == val], return_counts=True)
        hist = u_c[1] / np.sum(u_c[1])

        i = np.searchsorted(histogram_x, u_c[0])
        m = len(u_c[0])
        #print(np.shape(h_arr))
        #print(np.shape(hist))
        h_arr[idx, i] = hist

    filename = os.path.splitext(os.path.basename(index_img))[0]
    print(np.shape(h_arr)[0])
    np.savez(out_path + '/' + filename + '_histogram.npz', h_arr)


def block_gabor(table_path):
    table_list = os.listdir(table_path)
    for filename in table_list:
        out_path = table_path + '/' + filename
        for fr in [4, 6]:
            mean_std = None

            for th in range(1, 9):
                g = np.load(out_path + '/gabor_fr_' + str(fr) + '_th_' + str(th) + '_mean_std.npz')['arr_0']
                if mean_std is None:
                    mean_std = g
                else:
                    mean_std = np.hstack((g, mean_std))

            gabor_mean_mean = np.mean(mean_std[:, ::2], axis=1)
            gabor_mean_std = np.std(mean_std[:, ::2], axis=1)
            gabor_std_mean = np.mean(mean_std[:, 1::2], axis=1)
            gabor_std_std = np.std(mean_std[:, 1::2], axis=1)
            gabor_statistics = np.vstack((gabor_mean_mean, gabor_mean_std, gabor_std_mean, gabor_std_std))
            gabor_statistics = gabor_statistics.T
            out_file = out_path + '/gabor_fr_' + str(fr) + '_mean_std.npz'
            print('gabor', np.shape(gabor_statistics)[0])
            np.savez(out_file, gabor_statistics)


def dsm_slop(dsm_path, index_path):
    tif_list = os.listdir(dsm_path)

    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            f_n = os.path.splitext(filename)[0]
            a = f_n.split('_')[-1]
            out_path = index_path + '/top_mosaic_09cm_' + a
            dataset, cols, rows, geotransform, space = open_tif(dsm_path + '/' + filename)
            dsm = dataset.ReadAsArray()
            www = 500
            gdal.Warp(out_path + '/dem.tif',
                      dataset,
                      width=cols // www,
                      height=rows // www,
                      resampleAlg='min')

            dataset = gdal.Open(out_path + '/dem.tif')
            dem = dataset.ReadAsArray()
            dem = scipy.ndimage.zoom(dem, [rows / (rows // www), cols / (cols // www)], order=1)

            aaaa = dsm - dem
            creat_tif(out_path + '/r_dsm.tif', aaaa, space, geotransform, 7)
            slop = np.gradient(aaaa)
            s = (slop[0] ** 2 + slop[1] ** 2) ** 0.5
            creat_tif(out_path + '/slop.tif', s, space, geotransform, 7)
            print('dsm_slop:' + filename)


def dsm_process(dsm_path, index_path):
    '''
    # 先做NDVI掩膜，去除植被高度，然后插值，再提取地形
    '''


    tif_list = os.listdir(dsm_path)

    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            f_n = 'top_mosaic_09cm_' + os.path.splitext(filename)[0].split('_')[-1]

            dataset, cols, rows, geotransform, space = open_tif(dsm_path + '/' + filename)
            dsm = dataset.ReadAsArray()

            dataset, cols, rows, geotransform, space = open_tif(index_path + '/' + f_n + '/i.tif')

            n = 500

            dem = np.zeros((rows // n, cols // n))

            c_index_range = [i for i in range(0, cols, int(cols / (cols // n)))]
            if len(c_index_range) == cols // n:
                c_index_range.append(cols)
            else:
                c_index_range[-1] = cols

            r_index_range = [i for i in range(0, rows, int(rows / (rows // n)))]
            if len(r_index_range) == rows // n:
                r_index_range.append(rows)
            else:
                r_index_range[-1] = rows


            for i in range(np.shape(dem)[0]):
                for j in range(np.shape(dem)[1]):
                    block = dsm[r_index_range[i]:r_index_range[i + 1], c_index_range[j]:c_index_range[j + 1]]
                    dem[i, j] = np.percentile(block, 5)

            dem = scipy.ndimage.zoom(dem, [rows / np.shape(dem)[0], cols / np.shape(dem)[1]],
                                     order=3, mode='nearest', prefilter=False, grid_mode=True)

            dsm = dsm - dem
            dsm = dsm.astype(np.float32)
            blur_dsm = cv2.bilateralFilter(src=dsm, d=9, sigmaColor=5, sigmaSpace=50)
            creat_tif(index_path + '/' + f_n + '/blur_dsm.tif', blur_dsm, space, geotransform, 7)

            k_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
            k_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
            slop_x = signal.convolve2d(blur_dsm, k_x, mode='same', boundary='symm')
            slop_y = signal.convolve2d(blur_dsm, k_y, mode='same', boundary='symm')

            slop_x = slop_x * 0.15 / 0.09
            slop_y = slop_y * 0.15 / 0.09

            slop = np.arctan(np.square(slop_y ** 2 + slop_x ** 2))

            '''
            slop = np.square(slop_y ** 2 + slop_x ** 2)
            slop = 1 / (1 + np.exp(-slop))
            '''


            creat_tif(index_path + '/' + f_n + '/slop.tif', slop, space, geotransform, 7)
            creat_tif(index_path + '/' + f_n + '/dem.tif', dem, space, geotransform, 7)

            dsm = (dsm - np.min(dsm)) / np.ptp(dsm) * 255
            dsm = dsm.astype(np.uint8)

            #blur_dsm = cv2.GaussianBlur(dsm, ksize=(7, 7), sigmaX=0, sigmaY=0)
            #blur_dsm = cv2.bilateralFilter(src=dsm, d=7, sigmaColor=50, sigmaSpace=50)



            slop = np.absolute(slop)
            slop = (slop - np.min(slop)) / np.ptp(slop) * 255
            slop = slop.astype(np.uint8)

            e_dsm = cv2.equalizeHist(dsm)

            e_slop = cv2.equalizeHist(slop)
            e_slop = e_slop // 8
            e_slop = e_slop.astype(np.uint8)

            creat_tif(index_path + '/' + f_n + '/e_dsm.tif', e_dsm, space, geotransform)
            creat_tif(index_path + '/' + f_n + '/e_slop.tif', e_slop, space, geotransform)
            print(filename)


# 归一化图像
def normaliz(img_path, out_path, max_value):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    tif_list = os.listdir(img_path)

    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            f_n = os.path.splitext(filename)[0]
            if not os.path.exists(out_path + '/' + f_n):
                os.mkdir(out_path + '/' + f_n)

            dataset, cols, rows, geotransform, space = open_tif(img_path + '/' +filename)
            img = dataset.ReadAsArray()
            i = (img[0, :, :].astype(np.uint32) ** 2 + img[1, :, :].astype(np.uint32) ** 2 + img[2, :, :].astype(np.uint32) ** 2) ** 0.5
            #i = np.floor(i).astype(np.uint8)

            #plt.imshow(i)
            #plt.show()* 0.57735
            i[i < 1] = 1

            n_img_0 = np.floor(img[0, :, :].astype(np.float64) / i * max_value).astype(np.uint8)
            n_img_1 = np.floor(img[1, :, :].astype(np.float64) / i * max_value).astype(np.uint8)
            n_img_2 = np.floor(img[2, :, :].astype(np.float64) / i * max_value).astype(np.uint8)
            color = np.array([n_img_0, n_img_1, n_img_2])

            i = (img[0, :, :].astype(np.uint32) + img[1, :, :].astype(np.uint32) + img[2, :, :].astype(np.uint32)) / 3
            i = i.astype(np.uint8)
            creat_tif(out_path + '/' + f_n + '/i.tif', i, space, geotransform)
            creat_tif(out_path + '/' + f_n + '/r.tif', color[0], space, geotransform)
            creat_tif(out_path + '/' + f_n + '/g.tif', color[1], space, geotransform)
            creat_tif(out_path + '/' + f_n + '/b.tif', color[2], space, geotransform)
            creat_tif(out_path + '/' + f_n + '/color.tif', color, space, geotransform)
            print('normaliz', filename)


# rgb to cielab
def rgb_cie(img_path, out_path):

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    tif_list = os.listdir(img_path)

    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            f_n = os.path.splitext(filename)[0]
            if not os.path.exists(out_path + '/' + f_n):
                os.mkdir(out_path + '/' + f_n)

            dataset, cols, rows, geotransform, space = open_tif(img_path + '/' + filename)

            bgr = cv2.imread(img_path + '/' + filename)
            cielab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
            print(np.array(cielab))
            print(np.dtype(cielab[0, 0, 0]))
            print(cielab[:3, :3, :3])

            cie_array = np.array([cielab[:, :, 0],
                                  cielab[:, :, 1],
                                  cielab[:, :, 2], ])

            creat_tif(out_path + '/' + f_n + '/cielab.tif', cie_array, space, geotransform)


# rgb
def rgb_i(img_path, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)


    tif_list = os.listdir(img_path)

    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            f_n = os.path.splitext(filename)[0]
            if not os.path.exists(out_path + '/' + f_n):
                os.mkdir(out_path + '/' + f_n)

            dataset, cols, rows, geotransform, space = open_tif(img_path + '/' + filename)

            r = dataset.GetRasterBand(1).ReadAsArray() / 255 * 51
            g = dataset.GetRasterBand(2).ReadAsArray() / 255 * 51
            b = dataset.GetRasterBand(3).ReadAsArray() / 255 * 51
            i = np.mean(np.array([dataset.GetRasterBand(1).ReadAsArray(),
                                  dataset.GetRasterBand(2).ReadAsArray(),
                                  dataset.GetRasterBand(3).ReadAsArray()]), axis=0)

            creat_tif(out_path + '/' + f_n + '/r.tif', r, space, geotransform)
            creat_tif(out_path + '/' + f_n + '/g.tif', g, space, geotransform)
            creat_tif(out_path + '/' + f_n + '/b.tif', b, space, geotransform)
            creat_tif(out_path + '/' + f_n + '/i.tif', i, space, geotransform)


# LBP
def LBP(index_path):

    list = os.listdir(index_path)

    for filename in list:
        i_path = index_path + '/' + filename + '/i.tif'
        output = index_path + '/' + filename
        dataset, cols, rows, geotransform, space = open_tif(i_path)
        img = cv2.imread(i_path, 0)
        lbp = local_binary_pattern(img, 8, 1, method='ror')
        output_lbp = output + '/lbp1.tif'
        creat_tif(output_lbp, lbp, space, geotransform)

        lbp = local_binary_pattern(img, 8, 3, method='ror')
        output_lbp = output + '/lbp3.tif'
        creat_tif(output_lbp, lbp, space, geotransform)

        lbp = local_binary_pattern(img, 8, 5, method='ror')
        output_lbp = output + '/lbp5.tif'
        creat_tif(output_lbp, lbp, space, geotransform)
        print('LBP:', filename)


# 植被指数
def ndvi(img_path, index_path):

    tif_list = os.listdir(img_path)

    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            f_n = os.path.splitext(filename)[0]
            if not os.path.exists(index_path + '/' + f_n):
                os.mkdir(index_path  + '/' + f_n)

            i_path = img_path + '/' + filename
            output_path = index_path + '/' + f_n + '/ndvi.tif'
            dataset, cols, rows, geotransform, space = open_tif(i_path)
            nir = dataset.GetRasterBand(1).ReadAsArray().astype(np.float64)
            r = dataset.GetRasterBand(2).ReadAsArray().astype(np.float64)

            nd = (nir - r) / (nir + r)

            # print('com:', type(com[0, 0]))
            creat_tif(output_path, nd, space, geotransform, 7)


# COM指数
def COM(img_path, output_path):
    dataset, cols, rows, geotransform, space = open_tif(img_path)
    rgb = cv2.imread(img_path)

    s = rgb.sum(axis=2)
    r = np.true_divide(rgb[:, :, 0], s, where=s != 0)
    g = np.true_divide(rgb[:, :, 1], s, where=s != 0)
    b = np.true_divide(rgb[:, :, 2], s, where=s != 0)

    c = r ** 0.667 * b ** (1 - 0.667)
    exg = 2 * g - r - b
    cive = 0.441 * r - 0.811 * g + 0.385 * b + 18.78745
    veg = np.true_divide(g, c, out=np.zeros_like(c), where=c != 0)
    com = 0.36 * exg + 0.47 * cive + 0.17 * veg

    # print('com:', type(com[0, 0]))
    creat_tif(output_path + '/com.tif', com, space, geotransform, 7)


# gabor
def gabor_image(index_path):
    list = os.listdir(index_path)

    for filename in list:
        i_path = index_path + '/' + filename + '/i.tif'
        output = index_path + '/' + filename

        dataset, cols, rows, geotransform, space = open_tif(i_path)

        img = imread(i_path)
        gray = rgb2gray(img)

        a = np.pi / 9

        for fr in [4, 6]:
            gaber_img = []
            for th in range(1, 9):
                filt_real, filt_imag = gabor(gray, frequency=fr * 0.1, theta=th * a, bandwidth=1, sigma_x=None,
                                             sigma_y=None, n_stds=3, offset=0, mode='reflect', cval=0)

                creat_tif(output + '/gabor_fr_' + str(fr) + '_th_' + str(th) + '.tif', filt_real, space, geotransform,
                                7)
                gaber_img.append(filt_real)

            gaber_img = np.array(gaber_img)
            print(np.shape(gaber_img))
            max_gaber_img = np.max(gaber_img, axis=0)
            print(np.shape(max_gaber_img))
            creat_tif(output + '/gabor_fr_' + str(fr) + '_max.tif', max_gaber_img, space, geotransform, )


# 植被掩膜

def COM_MASK(com_table_path, segment_path, com_mask_path):

    com_mean = np.load(com_table_path)['arr_0'][:, 1]

    segment_dataset, cols, rows, geotransform, space = open_tif(segment_path)
    segment_array = segment_dataset.GetRasterBand(1).ReadAsArray()

    com_mask = np.zeros_like(segment_array)
    for x in range(np.shape(com_mask)[0]):
        for y in range(np.shape(com_mask)[1]):
            com_mask[x, y] = com_mean[segment_array[x, y]]

    creat_tif(com_mask_path, com_mask, space, geotransform, 7)


# 自适应阴影景观
def fuzzy_shadow_landscape3(index_path, degree, influence_length):
    list = os.listdir(index_path)

    for filename in list:
        i_path = index_path + '/' + filename + '/i.tif'
        r_path = index_path + '/' + filename + '/ndvi.tif'
        output = index_path + '/' + filename
        dataset, cols, rows, geotransform, space = open_tif(i_path)
        i = dataset.ReadAsArray()
        dataset, _, _, _, _ = open_tif(r_path)
        r = dataset.ReadAsArray()

        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(i)[0]
        shadow_lines = []
        line_img = np.zeros_like(i, dtype=np.uint8)
        for l in lines:
            l = l[0]
            rr, cc = draw.line(math.floor(l[1]), math.floor(l[0]), math.floor(l[3]), math.floor(l[2]))
            draw.set_color(line_img, [rr, cc], [255])

        creat_tif(output + '/line_img.tif', line_img, space, geotransform)

        line_shadow = np.zeros_like(i, dtype=np.uint8)
        f_s_l = np.zeros_like(i, dtype=np.uint8)
        for l in lines:
            l = l[0]
            dx = l[0] - l[2]
            dy = l[1] - l[3]
            # 排除长度短的
            if (dx ** 2 + dy ** 2) ** 0.5 < 70 or (dx ** 2 + dy ** 2) ** 0.5 > 600:
                continue

            # - pi / 2 < a <= 3 / 2 * pi [-90, 270]
            if dx == 0:
                a = math.pi / 2
            else:
                a = math.atan(dy / dx)

            # 排除和光照角度夹角小的
            if degree > 90:
                d = degree - 180
            else:
                d = degree
            if not 30 < abs(d - (a / math.pi * 180)) < 150:
                continue
            # 排除阴影侧阴影少的
            if a >= 0:
                s_d = a - math.pi / 2
            else:
                s_d = a + math.pi / 2
            width = 30
            wx = width * math.cos(s_d)
            wy = width * math.sin(s_d)
            if abs(s_d - degree) < math.pi / 2:
                wx = -wx
                wy = -wy

            mask = np.zeros_like(i, dtype=np.uint8)
            X = np.array([l[0], l[2], l[2] + wx, l[2] + wx + dx])
            Y = np.array([l[1], l[3], l[3] + wy, l[3] + wy + dy])
            X = np.around(X)
            Y = np.around(Y)
            rr, cc = draw.polygon(Y, X)

            draw.set_color(mask, [rr, cc], [255])
            m_sun = cv2.mean(i, mask)
            m_r = cv2.mean(r, mask)

            mask = np.zeros_like(shadow, dtype=np.uint8)
            X = np.array([l[0], l[2], l[2] - wx, l[2] - wx + dx])
            Y = np.array([l[1], l[3], l[3] - wy, l[3] - wy + dy])
            X = np.around(X)
            Y = np.around(Y)
            rr, cc = draw.polygon(Y, X)
            draw.set_color(line_shadow, [rr, cc], [255])
            draw.set_color(mask, [rr, cc], [255])
            m_shadow = cv2.mean(i, mask)
            #print(m)
            if m_shadow[0] > 55 or m_sun[0] < 80 or m_r[0] > 0.2:
                continue
            # 阴影影响
            wx = influence_length * math.cos(s_d)
            wy = influence_length * math.sin(s_d)
            X = np.array([l[0], l[2], l[2] + wx, l[2] + wx + dx])
            Y = np.array([l[1], l[3], l[3] + wy, l[3] + wy + dy])
            X = np.around(X)
            Y = np.around(Y)

            rr, cc = draw.polygon(Y, X)
            draw.set_color(f_s_l, [rr, cc], [255])

        creat_tif(output + '/line_shadow.tif', line_shadow, space, geotransform)
        creat_tif(output + '/fuzzy_shadow_landscape.tif', f_s_l, space, geotransform)
        print('fuzzy_shadow_landscape3:', filename)


'''
def shadow_landscape_org(index_path, degree, influence_length):
    list = os.listdir(index_path)

    for filename in list:
        i_path = index_path + '/' + filename + '/i.tif'
        r_path = index_path + '/' + filename + '/ndvi.tif'
        output = index_path + '/' + filename
        dataset, cols, rows, geotransform, space = open_tif(i_path)
        gray = dataset.ReadAsArray()
        dataset, _, _, _, _ = open_tif(r_path)
        r = dataset.ReadAsArray()

        # 阴影检测
        sh_arr = np.zeros_like(gray, np.uint8)
        sh_arr[gray > 65] = 255

        sh_arr = morphology.remove_small_objects(sh_arr, 2000)
        sh_arr = morphology.remove_small_holes(sh_arr, 2000)
        sh_arr = sh_arr.astype(np.uint8)
        sh_arr[sh_arr == 1] = 255
        tools.creat_tif(out_path + '/shadow.tif', sh_arr, space, geotransform)

        # 阴影景观
        i = np.copy(sh_arr)
        SE_intermediate = np.zeros([61, 61], dtype=np.uint8)
        SE_intermediate[30, 0:30] = 1
        SE_intermediate = rotate(SE_intermediate, -degree, order=0, preserve_range=True).astype('uint8')
        er = morphology.dilation(i, selem=SE_intermediate)
        # tools.creat_tif(out_path + '/er.tif', er, space, geotransform)
        di = er
        for _ in range(32):
            fp = morphology.disk(1)
            di = morphology.erosion(di, selem=fp)
            di[i > di] = i[i > di]
        shadow_landscape = morphology.erosion(di, selem=SE_intermediate)
        shadow_landscape = shadow_landscape - di
        shadow_landscape[shadow_landscape == 255] = 0
        shadow_landscape[r > 0.2] = 0
        shadow_landscape = shadow_landscape.astype(bool)

        shadow_landscape = morphology.remove_small_objects(shadow_landscape, 2000)
        shadow_landscape = shadow_landscape.astype(np.uint8)
        shadow_landscape[shadow_landscape == 1] = 255

        creat_tif(output + '/shadow_landscape_org.tif', shadow_landscape, space, geotransform)
'''

# 决策树分类 (建筑物label，植被label) (附带精度评价和样本数量)
def decision_tree(data_path):

    f_l_s_data = np.load(data_path + '\\fuzzy_shadow_landscape_mean_std.npz')

    decision_tree_label = (f_l_s_data['arr_0'][:, 0] > 200)

    decision_tree_label = decision_tree_label.astype(np.uint8)

    np.savez(data_path + "\\decision_tree_label.npz", decision_tree_label)


# pu-learning
#########################################################
def init(lock, u_data, p_data, sample_label, pre_proba, pre_n,):
    global lock_g, u_data_g, p_data_g, sample_label_g, pre_proba_g, pre_n_g
    lock_g = lock
    u_data_g = u_data
    p_data_g = p_data
    sample_label_g = sample_label
    pre_proba_g = pre_proba
    pre_n_g = pre_n


def trainProcess(l_p):
    NU, NP, N, ip = l_p

    aa = 1
    #number = 5000
    u_data = np.frombuffer(u_data_g.get_obj(), dtype=np.float64)
    u_data = u_data.reshape((-1, N))
    p_data = np.frombuffer(p_data_g.get_obj(), dtype=np.float64)
    p_data = p_data.reshape((-1, N))

    sample_label = np.array([1 for _ in range(NP)] + [0 for _ in range(NP)])
    pre_proba = np.frombuffer(pre_proba_g.get_obj(), dtype=np.float64)
    pre_n = np.frombuffer(pre_n_g.get_obj(), dtype=np.int)

    bootstrap_sample = np.random.choice(NU, NP, replace=False)
    u_sample = u_data[bootstrap_sample]
    data_bootstrap = np.vstack((p_data, u_sample))

    '''
    model = svm.SVC(C=10, gamma='scale', probability=True)
    model.fit(data_bootstrap, sample_label)
    idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
    p = model.predict_proba(u_data[idx_oob])
    p = p[:, 1]
    '''
    '''
    model = MLPClassifier(hidden_layer_sizes=(200), activation='logistic', alpha=100)
    model.fit(data_bootstrap, sample_label)
    idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
    p = model.predict(u_data[idx_oob])
    '''


    #class_weight = {0: 1, 1: 4}
    model = DecisionTreeClassifier(criterion='entropy', max_depth=30, max_features='auto')
    model.fit(data_bootstrap, sample_label)
    idx_oob = sorted(set(range(NU)) - set(np.unique(bootstrap_sample)))
    p = model.predict_proba(u_data[idx_oob])[:, 1]

    # 获取锁，用于进程同步
    lock_g.acquire()
    # 写入推理结果
    pre_proba[idx_oob] += p
    pre_n[idx_oob] += 1
    # 释放锁，开启下一个进程
    lock_g.release()
    print(str(ip) + 'write over', flush=True)
    return 0


def cross_validation(n_sample, p_sample):
    p_n = np.shape(p_sample)[0]
    bootstrap_sample = np.random.choice(np.shape(n_sample)[0], p_n, replace=False)
    n_sample = n_sample[bootstrap_sample]
    X = np.vstack((p_sample, n_sample))

    y = np.array([1 for _ in range(p_n)] + [0 for _ in range(p_n)])

    print(np.shape(p_sample))
    print(np.shape(n_sample))
    print(np.shape(X))
    print(np.shape(y))

    for i in np.arange(0.1, 1, 0.1):
        model = svm.SVC(C=i, probability=True)
        accs = model_selection.cross_val_score(model, X, y=y, scoring='f1', cv=10, n_jobs=1)
        print('c=%d十折交叉验证结果:' % i, np.mean(accs), np.std(accs))


#########################################################
def pu_learing1(table_path, f_name, ite, Parallel_number):
    tif_list = os.listdir(table_path)
    table_number = np.zeros_like(tif_list, np.int)
    X = None
    y = None
    real = None

    for file_index, filename in enumerate(tif_list):
        img_index = None
        for i in f_name:
            index = np.load(table_path + '/' + filename + '/' + i + '.npz')['arr_0']
            print(table_path + '/' + filename + '/' + i + '.npz')
            print(np.shape(index))
            if img_index is None:
                img_index = index
            else:
                img_index = np.hstack((img_index, index))


        if X is None:
            X = img_index
        else:
            X = np.vstack((X, img_index))

        d_t_l = np.load(table_path + '/' + filename + '/' + 'decision_tree_label.npz')['arr_0']
        real_a = np.load(table_path + '/' + filename + '/' + 'reallabel_mean_std.npz')['arr_0']
        if y is None:
            y = d_t_l
        else:
            y = np.hstack((y, d_t_l))

        if real is None:
            real = real_a[:, 0].flatten()
        else:
            y = np.hstack((y, d_t_l))

        table_number[file_index] = np.shape(img_index)[0]

    # 标准化
    print('标准化')
    print(np.dtype(X[0, 0]))
    print(np.shape(X))

    #X = pca.fit_transform(X)

    X = preprocessing.scale(X)

    print(np.shape(X)[1])

    N = np.shape(X)[1]

    # pu-learing
    NP = np.shape(X[y == 1])[0]
    NU = np.shape(X[y == 0])[0]
    print('正样本数：%d' % NP)
    print('负样本数：%d' % NU)

    lock = multiprocessing.Lock()
    u_data = multiprocessing.Array('d', list(X[y == 0].flatten()))
    p_data = multiprocessing.Array('d', list(X[y == 1].flatten()))
    sample_label = multiprocessing.Array('i', [1 for _ in range(NP)] + [0 for _ in range(NP)])

    pre_proba = multiprocessing.Array('d', [0 for _ in range(NU)])
    pre_n = multiprocessing.Array('i', [0 for _ in range(NU)])

    # pool = multiprocessing.Pool(processes=3, initializer=init, initargs=(lock, u_data, p_data, sample_label, pre_proba, pre_n,))

    with multiprocessing.Pool(processes=Parallel_number, initializer=init,
                              initargs=(lock, u_data, p_data, sample_label, pre_proba, pre_n,)) as pool:
        result = list(tqdm(pool.imap(trainProcess, [[NU, NP, N, ip] for ip in range(ite)]), total=ite))

    pool.close()
    pool.join()

    pre_proba_arr = np.frombuffer(pre_proba.get_obj(), dtype=np.float64)
    pre_n_arr = np.frombuffer(pre_n.get_obj(), dtype=np.int)

    pre_label = pre_proba_arr / pre_n_arr

    t_path = os.path.abspath(os.path.join(table_path, ".."))
    np.savez(t_path + '\\pre_label.npz', pre_label)

    y = y.astype(np.float64)
    y[y == 0] = pre_label

    for file_index, filename in enumerate(tif_list):
        if file_index == 0:
            start = 0
        else:
            start = table_number[:file_index].sum()
        end = start + table_number[file_index]
        np.savez(table_path + '/' + filename + '/' + 'result.npz', y[start:end])

####################################################################################################################


# OCSVM
def train_OCSVM_rbf(table_path, f_name):
    tif_list = os.listdir(table_path)
    table_number = np.zeros_like(tif_list, dtype=np.uint32)
    X = None
    y = None
    for file_index, filename in enumerate(tif_list):
        img_index = None
        for i in f_name:
            index = np.load(table_path + '/' + filename + '/' + i + '.npz')['arr_0']
            if img_index is None:
                img_index = index
            else:
                img_index = np.hstack((img_index, index))

        if X is None:
            X = img_index
        else:
            X = np.vstack((X, img_index))

        d_t_l = np.load(table_path + '/' + filename + '/' + 'decision_tree_label.npz')['arr_0']
        if y is None:
            y = d_t_l
        else:
            y = np.hstack((y, d_t_l))

        table_number[file_index] = np.shape(img_index)[0]

    # 标准化
    print('标准化')
    print(np.dtype(X[0, 0]))
    pca = PCA(n_components=0.99)
    # scaler = preprocessing.scale()
    pca = PCA(n_components=0.99)
    X_rgb = pca.fit_transform(X[:, 0:768])
    X_rgb = preprocessing.scale(X_rgb)

    X_gaber = preprocessing.scale(X[:, 768:780])

    X_lbp = pca.fit_transform(X[:, 780:888])
    X_lbp = preprocessing.scale(X_lbp)

    X_slop = preprocessing.scale(X[:, 888:920])

    X_dsm = preprocessing.scale(X[:, 920:922])

    X = np.hstack((X_rgb, X_gaber, X_lbp, X_slop, X_dsm))

    NP = np.shape(X[y == 1])[0]
    NU = np.shape(X[y == 0])[0]
    print('正样本数：%d' % NP)
    print('负样本数：%d' % NU)

    print(np.shape(X))
    print(np.shape(y))

    #############################################################
    g = np.shape(X)[1] * 0.01
    clf = svm.OneClassSVM(kernel='rbf', gamma=g)
    clf.fit(X[y == 1])

    pre_label = clf.predict(X[y == 0])

    ##############################################################

    y[y == 0] = pre_label

    for file_index, filename in enumerate(tif_list):
        if file_index == 0:
            start = 0
        else:
            start = table_number[:file_index].sum()
        end = start + table_number[file_index]
        np.savez(table_path + '/' + filename + '/' + 'result.npz', y[start:end])


def train_OCSVM_poly(table_path, f_name):
    tif_list = os.listdir(table_path)
    table_number = np.zeros_like(tif_list, dtype=np.uint32)
    X = None
    y = None
    for file_index, filename in enumerate(tif_list):
        img_index = None
        for i in f_name:
            index = np.load(table_path + '/' + filename + '/' + i + '.npz')['arr_0']
            if img_index is None:
                img_index = index
            else:
                img_index = np.hstack((img_index, index))

        if X is None:
            X = img_index
        else:
            X = np.vstack((X, img_index))

        d_t_l = np.load(table_path + '/' + filename + '/' + 'decision_tree_label.npz')['arr_0']
        if y is None:
            y = d_t_l
        else:
            y = np.hstack((y, d_t_l))

        table_number[file_index] = np.shape(img_index)[0]


    # 标准化
    X = preprocessing.scale(X)

    N = np.shape(X)[1]

    # pu-learing
    NP = np.shape(X[y == 1])[0]
    NU = np.shape(X[y == 0])[0]
    print('正样本数：%d' % NP)
    print('负样本数：%d' % NU)

    #############################################################
    pca = PCA(n_components=0.99)
    X_rgb = pca.fit_transform(X[:, 0:768])
    X_rgb = preprocessing.scale(X_rgb)

    X_gaber = preprocessing.scale(X[:, 768:780])

    X_lbp = pca.fit_transform(X[:, 780:888])
    X_lbp = preprocessing.scale(X_lbp)

    X_slop = preprocessing.scale(X[:, 888:920])

    X_dsm = preprocessing.scale(X[:, 920:922])

    X = np.hstack((X_rgb, X_gaber, X_lbp, X_slop, X_dsm))

    clf = svm.OneClassSVM(nu=0.1, kernel='poly', gamma=0.1)
    clf.fit(X[y == 1])

    pre_label = clf.predict(X[y == 0])

    ##############################################################
    y[y == 0] = pre_label

    for file_index, filename in enumerate(tif_list):
        if file_index == 0:
            start = 0
        else:
            start = table_number[:file_index].sum()
        end = start + table_number[file_index]
        np.savez(table_path + '/' + filename + '/' + 'result.npz', y[start:end])


def train_OCSVM_sig(table_path, f_name):
    tif_list = os.listdir(table_path)
    table_number = np.zeros_like(tif_list, dtype=np.uint32)
    X = None
    y = None
    for file_index, filename in enumerate(tif_list):
        img_index = None
        for i in f_name:
            index = np.load(table_path + '/' + filename + '/' + i + '.npz')['arr_0']
            if img_index is None:
                img_index = index
            else:
                img_index = np.hstack((img_index, index))

        if X is None:
            X = img_index
        else:
            X = np.vstack((X, img_index))

        d_t_l = np.load(table_path + '/' + filename + '/' + 'decision_tree_label.npz')['arr_0']
        if y is None:
            y = d_t_l
        else:
            y = np.hstack((y, d_t_l))

        table_number[file_index] = np.shape(img_index)[0]


    # 标准化
    X = preprocessing.scale(X)

    N = np.shape(X)[1]

    # pu-learing
    NP = np.shape(X[y == 1])[0]
    NU = np.shape(X[y == 0])[0]
    print('正样本数：%d' % NP)
    print('负样本数：%d' % NU)

    #############################################################
    pca = PCA(n_components=0.99)
    X_rgb = pca.fit_transform(X[:, 0:768])
    X_rgb = preprocessing.scale(X_rgb)

    X_gaber = preprocessing.scale(X[:, 768:780])

    X_lbp = pca.fit_transform(X[:, 780:888])
    X_lbp = preprocessing.scale(X_lbp)

    X_slop = preprocessing.scale(X[:, 888:920])

    X_dsm = preprocessing.scale(X[:, 920:922])

    X = np.hstack((X_rgb, X_gaber, X_lbp, X_slop, X_dsm))

    clf = svm.OneClassSVM(nu=0.1, kernel='sigmoid', gamma=0.1)
    clf.fit(X[y == 1])

    pre_label = clf.predict(X[y == 0])

    ##############################################################


    y[y == 0] = pre_label

    for file_index, filename in enumerate(tif_list):
        if file_index == 0:
            start = 0
        else:
            start = table_number[:file_index].sum()
        end = start + table_number[file_index]
        np.savez(table_path + '/' + filename + '/' + 'result.npz', y[start:end])


# KDE
from sklearn.neighbors import KernelDensity
def train_KDE(table_path, f_name):
    tif_list = os.listdir(table_path)
    table_number = np.zeros_like(tif_list, dtype=np.uint32)
    X = None
    y = None
    for file_index, filename in enumerate(tif_list):
        img_index = None
        for i in f_name:
            index = np.load(table_path + '/' + filename + '/' + i + '.npz')['arr_0']
            if img_index is None:
                img_index = index
            else:
                img_index = np.hstack((img_index, index))

        if X is None:
            X = img_index
        else:
            X = np.vstack((X, img_index))

        d_t_l = np.load(table_path + '/' + filename + '/' + 'decision_tree_label.npz')['arr_0']
        if y is None:
            y = d_t_l
        else:
            y = np.hstack((y, d_t_l))

        table_number[file_index] = np.shape(img_index)[0]


    # 标准化
    X = preprocessing.scale(X)

    pca = PCA(n_components=0.9)
    X = pca.fit_transform(X)
    print('X:', np.shape(X))

    N = np.shape(X)[1]

    # pu-learing
    NP = np.shape(X[y == 1])[0]
    NU = np.shape(X[y == 0])[0]
    print('正样本数：%d' % NP)
    print('负样本数：%d' % NU)

    #############################################################
    pca = PCA(n_components=0.99)
    X_rgb = pca.fit_transform(X[:, 0:768])
    X_rgb = preprocessing.scale(X_rgb)

    X_gaber = preprocessing.scale(X[:, 768:780])

    X_lbp = pca.fit_transform(X[:, 780:888])
    X_lbp = preprocessing.scale(X_lbp)

    X_slop = preprocessing.scale(X[:, 888:920])

    X_dsm = preprocessing.scale(X[:, 920:922])

    X = np.hstack((X_rgb, X_gaber, X_lbp, X_slop, X_dsm))

    pca = PCA(n_components=10)
    X = pca.fit_transform(X)

    kde0 = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(X[y == 0])
    kde1 = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(X[y == 1])

    log_density0 = kde0.score_samples(X[y == 0])
    log_density1 = kde1.score_samples(X[y == 0])

    score = log_density1 / log_density0

    pre_label = np.zeros_like(score)
    pre_label[log_density >= 0] = 1 / (1 + np.exp(-log_density[log_density >= 0]))
    pre_label[log_density < 0] = np.exp(log_density[log_density < 0]) / (1 + np.exp(log_density[log_density < 0]))

    print('pre:', np.min(pre_label))
    print('pre:', np.max(pre_label))
    print('pre:', np.mean(pre_label))
    print('pre:', np.std(pre_label))

    #pre_label = preprocessing.scale(pre_label)
    ##############################################################

    y[y == 0] = pre_label

    for file_index, filename in enumerate(tif_list):
        if file_index == 0:
            start = 0
        else:
            start = table_number[:file_index].sum()
        end = start + table_number[file_index]
        np.savez(table_path + '/' + filename + '/' + 'result.npz', y[start:end])


# IsolationForest
from sklearn.ensemble import IsolationForest
def train_Iso(table_path, f_name):
    tif_list = os.listdir(table_path)
    table_number = np.zeros_like(tif_list, dtype=np.uint32)
    X = None
    y = None
    for file_index, filename in enumerate(tif_list):
        img_index = None
        for i in f_name:
            index = np.load(table_path + '/' + filename + '/' + i + '.npz')['arr_0']
            if img_index is None:
                img_index = index
            else:
                img_index = np.hstack((img_index, index))

        if X is None:
            X = img_index
        else:
            X = np.vstack((X, img_index))

        d_t_l = np.load(table_path + '/' + filename + '/' + 'decision_tree_label.npz')['arr_0']
        if y is None:
            y = d_t_l
        else:
            y = np.hstack((y, d_t_l))

        table_number[file_index] = np.shape(img_index)[0]


    # 标准化
    X = preprocessing.scale(X)

    N = np.shape(X)[1]

    # pu-learing
    NP = np.shape(X[y == 1])[0]
    NU = np.shape(X[y == 0])[0]
    print('正样本数：%d' % NP)
    print('负样本数：%d' % NU)

    #############################################################
    pca = PCA(n_components=0.90)
    X_rgb = pca.fit_transform(X[:, 0:768])
    X_rgb = preprocessing.scale(X_rgb)

    X_gaber = preprocessing.scale(X[:, 768:780])

    X_lbp = pca.fit_transform(X[:, 780:888])
    X_lbp = preprocessing.scale(X_lbp)

    X_slop = preprocessing.scale(X[:, 888:920])

    X_dsm = preprocessing.scale(X[:, 920:922])

    X = np.hstack((X_rgb, X_gaber, X_lbp, X_slop, X_dsm))

    pca = PCA(n_components=20)
    X = pca.fit_transform(X)

    X = np.hstack((X_slop, X_dsm))

    clf = IsolationForest(n_estimators=100, max_samples=0.9, contamination=0, ).fit(X[y == 1])
    #pre_label = clf.score_samples(X[y == 0])
    pre_label = clf.decision_function(X[y == 0])

    ##############################################################
    y[y == 0] = pre_label

    for file_index, filename in enumerate(tif_list):
        if file_index == 0:
            start = 0
        else:
            start = table_number[:file_index].sum()
        end = start + table_number[file_index]
        np.savez(table_path + '/' + filename + '/' + 'result.npz', y[start:end])


def train_S_EM(table_path, f_name, per):
    tif_list = os.listdir(table_path)
    table_number = np.zeros_like(tif_list, dtype=np.uint32)
    X = None
    y = None
    for file_index, filename in enumerate(tif_list):
        img_index = None
        for i in f_name:
            index = np.load(table_path + '/' + filename + '/' + i + '.npz')['arr_0']
            if img_index is None:
                img_index = index
            else:
                img_index = np.hstack((img_index, index))

        if X is None:
            X = img_index
        else:
            X = np.vstack((X, img_index))

        d_t_l = np.load(table_path + '/' + filename + '/' + 'decision_tree_label.npz')['arr_0']
        if y is None:
            y = d_t_l
        else:
            y = np.hstack((y, d_t_l))

        table_number[file_index] = np.shape(img_index)[0]

    # 标准化
    X = preprocessing.scale(X)

    N = np.shape(X)[1]

    # pu-learing
    NP = np.shape(X[y == 1])[0]
    NU = np.shape(X[y == 0])[0]
    print('正样本数：%d' % NP)
    print('负样本数：%d' % NU)

    #############################################################
    pca = PCA(n_components=0.99)
    X_rgb = pca.fit_transform(X[:, 0:768])
    X_rgb = preprocessing.scale(X_rgb)

    X_gaber = preprocessing.scale(X[:, 768:780])

    X_lbp = pca.fit_transform(X[:, 780:888])
    X_lbp = preprocessing.scale(X_lbp)

    X_slop = preprocessing.scale(X[:, 888:920])

    X_dsm = preprocessing.scale(X[:, 920:922])

    X = np.hstack((X_rgb, X_gaber, X_lbp, X_slop, X_dsm))
    ####################################################################
    y1 = y.copy()
    index = np.argwhere(y == 1).flatten()
    print('y==1where:', np.shape(index))
    b = np.random.choice(np.shape(index)[0], int(np.shape(index)[0] * 0.15), replace=False)
    print('', np.shape(b))
    S_index = index[b]
    y1[S_index] = 0

    clf = GaussianNB()
    clf.fit(X, y1)

    print('S_index', np.shape(S_index))
    print('X', np.shape(X))
    print('X[y == 0]', np.shape(X[y == 0]))
    print('X[S_index]', np.shape(X[S_index]))

    p = clf.predict_proba(X[y1 == 0])
    print('p', np.shape(p))
    p = p[:, 1]
    print('p[:, 1]', np.shape(p))

    p_s = clf.predict_proba(X[S_index])
    print('p_s', np.shape(p_s))
    p_s = p_s[:, 1]
    print('p_s[:, 1]', np.shape(p_s))

    th = np.percentile(p_s, per)
    #th = np.min(p_s)
    p[p > th] = 2
    #p[p > th] = 1
    p[p <= th] = 0
    y1[y1 == 0] = p
    y1[S_index] = 1

    clf = GaussianNB()
    clf.fit(X[y1 < 2], y1[y1 < 2])
    p = clf.predict(X[y1 == 2])
    y1[y1 == 2] = p

    for i in range(100):
        clf = GaussianNB()
        clf.fit(X, y1)
        y1 = clf.predict(X)
        y1[y == 1] = 1

    ##############################################################
    y = y1

    for file_index, filename in enumerate(tif_list):
        if file_index == 0:
            start = 0
        else:
            start = table_number[:file_index].sum()
        end = start + table_number[file_index]
        np.savez(table_path + '/' + filename + '/' + 'result.npz', y[start:end])


# SVDD
from BaseSVDD import BaseSVDD, BananaDataset
def train_SVDD_rbf(table_path, f_name, ite, Parallel_number):
    tif_list = os.listdir(table_path)
    table_number = np.zeros_like(tif_list, np.int)
    X = None
    y = None
    for file_index, filename in enumerate(tif_list):
        img_index = None
        for i in f_name:
            index = np.load(table_path + '/' + filename + '/' + i + '.npz')['arr_0']
            print(table_path + '/' + filename + '/' + i + '.npz')
            print(np.shape(index))
            if img_index is None:
                img_index = index
            else:
                img_index = np.hstack((img_index, index))

        if X is None:
            X = img_index
        else:
            X = np.vstack((X, img_index))

        d_t_l = np.load(table_path + '/' + filename + '/' + 'decision_tree_label.npz')['arr_0']
        if y is None:
            y = d_t_l
        else:
            y = np.hstack((y, d_t_l))

        table_number[file_index] = np.shape(img_index)[0]


    # 标准化
    print('标准化')
    print(np.dtype(X[0, 0]))
    print(np.shape(X))
    #pca = PCA(n_components=0.99)

    X = preprocessing.scale(X)

    print(np.shape(X)[1])

    N = np.shape(X)[1]

    NP = np.shape(X[y == 1])[0]
    NU = np.shape(X[y == 0])[0]
    print('正样本数：%d' % NP)
    print('负样本数：%d' % NU)

    #############################################################
    y_train = y[y == 1]
    X_train = X[y == 1]
    svdd = BaseSVDD(C=0.9, kernel='rbf', gamma=0.3, display='on')
    svdd.fit(X_train, y_train)
    y_pre = svdd.predict(X[y == 0])
    print(np.shape(y_pre))
    print(np.dtype(y_pre[0]))
    y[y == 0] = y_pre
    ########################################################

    t_path = os.path.abspath(os.path.join(table_path, ".."))
    np.savez(t_path + '\\pre_label.npz', y)
    ##################################################

    for file_index, filename in enumerate(tif_list):
        if file_index == 0:
            start = 0
        else:
            start = table_number[:file_index].sum()
        end = start + table_number[file_index]
        np.savez(table_path + '/' + filename + '/' + 'result.npz', y[start:end])



##########################################################
# 数据表转图
def data2tif(data_path, segment_path, out_path):
    label = np.load(data_path)['arr_0']
    label[label == -1] = 0

    segment_dataset, cols, rows, geotransform, space = open_tif(segment_path)
    se = segment_dataset.GetRasterBand(1).ReadAsArray()
    re = label[se]
    creat_tif(out_path, re, space, geotransform, 7)


# 结果评价
def evaluation(th_list, real_label_path, pre_path, out_path):
    dataset, cols, rows, geotransform, space = open_tif(real_label_path)
    real_arr = dataset.ReadAsArray()
    real_arr = (real_arr[0, :, :] == 0) & (real_arr[1, :, :] == 0) & (real_arr[2, :, :] == 255)
    #print(np.shape(real_arr))

    dataset, cols, rows, geotransform, space = open_tif(pre_path)
    pre_arr = dataset.GetRasterBand(1).ReadAsArray()

    true_positive = np.zeros(np.shape(th_list))
    true_negative = np.zeros(np.shape(th_list))
    false_positive = np.zeros(np.shape(th_list))
    false_negative = np.zeros(np.shape(th_list))

    f = open(out_path + '/data.txt', 'w')

    #for x in tqdm(range(np.shape(pre_arr)[0]), desc='评价'):


    for x in range(np.shape(pre_arr)[0]):
        for y in range(np.shape(pre_arr)[1]):
            pre = (pre_arr[x, y] > th_list)
            real = real_arr[x, y]

            true_positive += pre & real
            true_negative += ~(pre | real)
            false_positive += pre & ~real
            false_negative += ~pre & real


    
    # roc
    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (false_positive + true_negative)
    auc = metrics.auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC\nAUC = " + str(auc))
    plt.legend()

    plt.savefig(out_path + '\\roc.png')
    plt.cla()

    f.write('ROC\n')
    f.write('tpr,fpr\n')
    for i in range(len(tpr)):
        f.write(str(tpr[i]) + ',' + str(fpr[i]) + '\n')
    # MIoU
    m = true_positive / (true_positive + false_negative + false_positive)
    plt.plot(th_list, m, label="MIoU")
    plt.title('Mean Intersection over Union')
    plt.xlabel("Threshold")
    plt.legend()

    plt.savefig(out_path + '\\MIoU.png')
    plt.cla()

    f.write('MIoU\n')
    f.write('th_list,m\n')
    for i in range(len(th_list)):
        f.write(str(th_list[i]) + ',' + str(m[i]) + '\n')
    # precision&recall
    p = true_positive / (true_positive + false_positive)
    r = true_positive / (true_positive + false_negative)
    plt.plot(th_list, p, label="Precision")
    plt.plot(th_list, r, label="Recall")
    plt.title('Precision and Recall')
    plt.xlabel("Threshold")
    plt.legend()

    plt.savefig(out_path + '\\precision&recall.png')
    plt.cla()

    f.write('precision&recall\n')
    f.write('th_list,p,r\n')
    for i in range(len(th_list)):
        f.write(str(th_list[i]) + ',' + str(p[i]) + ',' + str(r[i]) + '\n')
    # f1
    f1 = 2 * p * r / (p + r)
    plt.plot(th_list, f1, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title('F1-score')
    plt.legend()

    plt.savefig(out_path + '\\f1.png')
    plt.cla()

    plt.close()

    f.write('f1\n')
    f.write('th_list,f1\n')
    for i in range(len(th_list)):
        f.write(str(th_list[i]) + ',' + str(f1[i]) + '\n')

    f.close()
    
    data = np.vstack((true_positive, true_negative, false_positive, false_negative))
    print(out_path)
    print('p:', p[0], 'r:', r[0])
    return data


# 后处理
def processing1(par):
    r_path, segment_path, out_path, t_path, img_path = par

    dataset, cols, rows, geotransform, space = open_tif(r_path)

    dataset, _, _, _, _ = open_tif(segment_path)
    segment_array = dataset.ReadAsArray()

    print(np.shape(np.unique(segment_array)))

    g = graph.RAG(segment_array)
    adj = adjacency_matrix(g).todense()
    adj = adj.astype(np.int8)
    print(np.shape(adj))

    result = np.load(t_path)['arr_0']
    result = result.astype(np.float16)
    one_arr = np.ones_like(result)
    print(np.shape(result))

    for i in range(5):
        c = np.matmul(result, adj).astype(np.float16)
        n = np.matmul(one_arr, adj).astype(np.int8)
        print(np.shape(c))
        #e = c / n
        e = c
        e = e.astype(np.float16)

        result = result.T
        e = e.A
        e = e.flatten()
        print(np.shape(e))
        result = result * 0.7 + e * 0.3
        print(np.shape(result))
        re = result[segment_array]
        ite = i + 1
        creat_tif(out_path + '/result_' + str(ite) + '.tif', re, space, geotransform, 7)
        print(out_path + '/result_' + str(ite) + '.tif')
    return 0



def processing2(par):
    r_path, segment_path, out_path, t_path, thr_min, thr_max, img_path = par

    dataset, cols, rows, geotransform, space = open_tif(r_path)
    # res = dataset.ReadAsArray()

    dataset, _, _, _, _ = open_tif(segment_path)
    segment_array = dataset.ReadAsArray()

    #print(np.shape(np.unique(segment_array)))

    g = graph.RAG(segment_array)
    adj = adjacency_matrix(g).todense()
    adj = adj.astype(np.int8)
    #print(np.shape(adj))

    result = np.load(t_path)['arr_0']

    sure = np.zeros_like(result)
    sure[result >= thr_max] = 1

    for i in range(3):
        c = np.matmul(sure, adj)
        c[0, result < thr_min] = 0
        c = c.A.flatten()
        print('c:', np.shape(c))

        sure[c > 0] = 1

    sure = sure.astype(np.uint8)
    #sure[sure == 1] = 255

    re = sure[segment_array]
    re = re.astype(np.bool)

    # 去除小斑块
    re = morphology.remove_small_objects(re, 2000)

    # 开操作去除毛刺
    kernal = morphology.disk(10)
    re = morphology.opening(re, kernal)

    # 闭操作去除空洞
    re = morphology.closing(re, kernal)


    # 去除空洞
    re = morphology.remove_small_objects(re, 500)

    re = re.astype(np.uint8)
    re[re == 1] = 255
    creat_tif(out_path + '/binarization.tif', re, space, geotransform)
    print(out_path)

def processing3(par):
    r_path, segment_path, out_path, t_path, thr_min, thr_max, img_path = par

    dataset, cols, rows, geotransform, space = open_tif(r_path)
    res = dataset.ReadAsArray()

    re = np.zeros_like(res)

    re[res >= 0.3] = 1

    re = re.astype(np.uint8)
    #sure[sure == 1] = 255


    re = re.astype(np.bool)

    # 去除小斑块
    re = morphology.remove_small_objects(re, 2000)

    # 开操作去除毛刺
    kernal = morphology.disk(10)
    re = morphology.opening(re, kernal)

    # 闭操作去除空洞
    re = morphology.closing(re, kernal)


    # 去除空洞
    re = morphology.remove_small_objects(re, 500)

    re = re.astype(np.uint8)
    re[re == 1] = 255
    creat_tif(out_path + '/binarization.tif', re, space, geotransform)
    print(out_path)

def processing4(par):
    r_path, segment_path, out_path, t_path, thr_min, thr_max, img_path = par

    dataset, cols, rows, geotransform, space = open_tif(r_path)
    # res = dataset.ReadAsArray()

    dataset, _, _, _, _ = open_tif(segment_path)
    segment_array = dataset.ReadAsArray()

    #print(np.shape(np.unique(segment_array)))

    g = graph.RAG(segment_array)
    adj = adjacency_matrix(g).todense()
    adj = adj.astype(np.uint8)
    #print(np.shape(adj))

    result = np.load(t_path)['arr_0']

    sure = np.zeros_like(result)
    sure[result >= thr_max] = 1

    # 二值化
    P = np.zeros_like(result)
    P[result >= 0.5] = 1

    # 基于图的开
    N = np.zeros_like(P)
    N[P == 0] = 1
    P[np.matmul(N, adj) > 0] = 0

    P = P + np.matmul(P, adj)
    P[P > 0] = 1
    # 基于图的闭
    P = P + np.matmul(P, adj)
    P[P > 0] = 1

    N = np.zeros_like(P)
    N[P == 0] = 1
    P[np.matmul(N, adj) > 0] = 0

    sure = sure.astype(np.uint8)
    #sure[sure == 1] = 255

    re = sure[segment_array]
    re = re.astype(np.bool)

    # 去除小斑块
    re = morphology.remove_small_objects(re, 2000)

    # 开操作去除毛刺
    kernal = morphology.disk(10)
    re = morphology.opening(re, kernal)

    # 闭操作去除空洞
    re = morphology.closing(re, kernal)


    # 去除空洞
    re = morphology.remove_small_objects(re, 500)

    re = re.astype(np.uint8)
    re[re == 1] = 255
    creat_tif(out_path + '/binarization.tif', re, space, geotransform)
    print(out_path)


def muti_processing1(result_path, img_path, segment_path, process2_path, table_path, index_path, thr_min, thr_max):
    tif_list = os.listdir(img_path)

    '''
    p = multiprocessing.Pool(1)
    res = []
    for tif in tif_list:
        if os.path.splitext(tif)[1] == '.tif':
            file = os.path.splitext(tif)[0]
            r_tif = result_path + '/' + file + '/result.tif'
            t_path = table_path + '/' + file + '/result.npz'
            s_tif = segment_path + '/' + file + '.tif'
            out_path = process_path + '/' + file
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            arg = [r_tif, s_tif, out_path, t_path]
            res.append(p.apply_async(processing1, args=(arg,)))

        #processing1(arg)
    p.close()
    p.join()
    print(res[0])
    '''
    for tif in tif_list:
        if os.path.splitext(tif)[1] == '.tif':
            file = os.path.splitext(tif)[0]
            r_tif = result_path + '/' + file + '/result.tif'
            t_path = table_path + '/' + file + '/result.npz'
            s_tif = segment_path + '/' + file + '.tif'
            i_path = index_path + '/' + file + '/line_img.tif'
            #out_path = process_path + '/' + file
            out_path = process2_path + '/' + file
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            arg = [r_tif, s_tif, out_path, t_path, thr_min, thr_max, img_path + '/' + tif, i_path]
            processing4(arg)

