##########################
# -*- coding: utf-8 -*-

import numpy as np
import tools
import sys
from skimage import morphology
import cv2
import math
import scipy


def find_value(arr, start, step, iter):
    iter += 1

    position = start + step
    if 0 <= position[0] and position[0] < np.shape(arr)[0] and 0 <= position[1] and position[1] < np.shape(arr)[1]:

        if arr[position[0], position[1]] != arr[position[0], position[1]]:
            return find_value(arr, start + step, step, iter)
        else:
            return arr[position[0], position[1]], iter
    else:
        return np.nan, iter


def inter_2D(arr):
    step1 = np.array([0, 1])
    step2 = np.array([0, -1])
    step3 = np.array([1, 0])
    step4 = np.array([-1, 0])

    output = np.copy(arr)
    w, h = np.shape(arr)
    for i in range(w):
        for j in range(h):

            if arr[i, j] != arr[i, j]:
                start = np.array([i, j])

                a1, i1 = find_value(arr, start, step1, 0)
                a2, i2 = find_value(arr, start, step2, 0)
                a3, i3 = find_value(arr, start, step3, 0)
                a4, i4 = find_value(arr, start, step4, 0)
                #print(i, j, a1, i1)
                value = []
                weight = []
                for v in np.array([[a1, i1], [a2, i2], [a3, i3], [a4, i4]]):
                    if v[0] == v[0]:
                        #print(v[0])
                        value.append(v[0])
                        weight.append(1 / v[1])

                output[i, j] = np.sum(np.array(value) * np.array(weight)) / np.sum(np.array(weight))
    return output


def inter_2D_test01(arr):
    print('max_arr:', np.max(arr))
    t = 1
    a = np.zeros((t * 2 + 1, t * 2 + 1))
    for i in range(t * 2 + 1):
        for j in range(t * 2 + 1):
            a[i, j] = 1 / (2 * math.pi) ** 0.5 / t * math.e ** (-((i - t) ** 2 + (j - t) ** 2) / 2 / t ** 2)
    a = np.round(a * 100)
    print(a)

    arr = cv2.copyMakeBorder(arr, t, t, t, t, borderType=cv2.BORDER_CONSTANT, value=0)
    w, h = np.shape(arr)

    while np.min(arr[t:w - t, t:h - t]) == 0:
        mask = np.zeros_like(arr)
        mask[arr != 0] = 1
        new_arr = np.copy(arr)
        for i in range(t, w - t):
            for j in range(t, h - t):
                if new_arr[i, j] == 0:
                    x1 = i - t
                    x2 = i + t + 1
                    y1 = j - t
                    y2 = j + t + 1

                    s = np.sum(mask[x1:x2, y1:y2] * a)
                    if s > 0:
                        new_arr[i, j] = np.sum(arr[x1:x2, y1:y2] * a) / s
                        if np.sum(arr[x1:x2, y1:y2] * a) / s > 100000:
                            print('s:', s)
                            print('arr:', arr[x1:x2, y1:y2].astype(np.uint8))
                            print('mask:', mask[x1:x2, y1:y2].astype(np.uint8))
                            print('arr*a:', (arr[x1:x2, y1:y2] * a).astype(np.uint8))
        arr = new_arr
        #print(111)
    return arr[t:w - t, t:h - t]


def building_mask(dsm, step, up, down):
    mask1 = np.zeros_like(dsm, np.uint8)
    for i in range(step, np.shape(dsm)[0], 1):
        grad = dsm[i, :] - dsm[i - step, :]
        g = np.zeros_like(grad)
        g[grad > up] = 1
        g[grad < down] = -1

        g[g == 0] = mask1[i - 1, :][g == 0]
        g[g == -1] = 0

        mask1[i, :] = g

    mask2 = np.zeros_like(dsm, np.uint8)
    for i in range(np.shape(dsm)[0] - step - 1, 0, -1):
        grad = dsm[i, :] - dsm[i + step, :]
        g = np.zeros_like(grad)
        g[grad > up] = 1
        g[grad < down] = -1

        g[g == 0] = mask2[i + 1, :][g == 0]
        g[g == -1] = 0

        mask2[i, :] = g

    mask3 = np.zeros_like(dsm, np.uint8)
    for i in range(step, np.shape(dsm)[1], 1):
        grad = dsm[:, i] - dsm[:, i - step]
        g = np.zeros_like(grad)
        g[grad > up] = 1
        g[grad < down] = -1

        g[g == 0] = mask3[:, i - 1][g == 0]
        g[g == -1] = 0

        mask3[:, i] = g

    mask4 = np.zeros_like(dsm, np.uint8)
    for i in range(np.shape(dsm)[1] - step - 1, 0, -1):
        grad = dsm[:, i] - dsm[:, i - step]
        g = np.zeros_like(grad)
        g[grad > up] = 1
        g[grad < down] = -1

        g[g == 0] = mask4[:, i - 1][g == 0]
        g[g == -1] = 0

        mask4[:, i] = g

    mask = mask1 + mask2 + mask3 + mask4
    print(np.max(mask))
    mask[mask > 1] = 1
    return mask






# 路径
DSM_path = r'D:\vaihingen_Automatic_extraction_of_buildings\结果\nDSM\1.tif'
realllabel_path = r'D:\vaihingen_Automatic_extraction_of_buildings\结果\nDSM\real.tif'
out_path = r'D:\vaihingen_Automatic_extraction_of_buildings\结果\nDSM'
ndvi_path = r'D:\vaihingen_Automatic_extraction_of_buildings\结果\nDSM\ndvi.tif'


dataset, cols, rows, geotransform, space = tools.open_tif(DSM_path)
dsm = dataset.ReadAsArray()


# 阶跃法
mask = building_mask(dsm, 5, 1, -1)
mask = mask.astype(np.uint8)

p_dsm = np.zeros_like(dsm, np.float64)
p_dsm[mask == 0] = dsm[mask == 0]
jie_dem = inter_2D_test01(p_dsm)

jie_mask_dsm = np.copy(dsm)
jie_mask_dsm[mask == 1] = 0
tools.creat_tif(out_path + '/jie_mask.tif', jie_mask_dsm, space, geotransform, 7)
print('2d')
# 输出此DEM
jie_ndsm = dsm - jie_dem
tools.creat_tif(out_path + '/jie_dem.tif', jie_dem, space, geotransform, 7)
tools.creat_tif(out_path + '/jie_ndsm.tif', jie_ndsm, space, geotransform, 7)


# 重采样法
s_dsm = np.copy(dsm)
s_dem = None
for _ in range(3):
    n = 500
    s_dem = np.zeros((rows // n, cols // n))
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

    for i in range(np.shape(s_dem)[0]):
        for j in range(np.shape(s_dem)[1]):
            block = s_dsm[r_index_range[i]:r_index_range[i + 1], c_index_range[j]:c_index_range[j + 1]]
            s_dem[i, j] = np.percentile(block, 5)

    s_dem = scipy.ndimage.zoom(s_dem, [rows / np.shape(s_dem)[0], cols / np.shape(s_dem)[1]],
                             order=3, mode='nearest', prefilter=False, grid_mode=True)

    s_ndsm = s_dsm - s_dem
    s_ndsm = s_ndsm.astype(np.float64)
tools.creat_tif(out_path + '/s_dem.tif', s_dem, space, geotransform, 7)
tools.creat_tif(out_path + '/s_ndsm.tif', s_ndsm, space, geotransform, 7)




realllabel_path = r'D:\vaihingen_Automatic_extraction_of_buildings\结果\nDSM\real.tif'
s_ndsm_path = r'D:\vaihingen_Automatic_extraction_of_buildings\结果\nDSM\s_ndsm.tif'
jie_ndsm_path = r'D:\vaihingen_Automatic_extraction_of_buildings\结果\nDSM\jie_ndsm.tif'

dataset, cols, rows, geotransform, space = tools.open_tif(DSM_path)
dsm = dataset.ReadAsArray()

dataset, _, _, _, _ = tools.open_tif(s_ndsm_path)
s_ndsm = dataset.ReadAsArray()

dataset, _, _, _, _ = tools.open_tif(jie_ndsm_path)
jie_ndsm = dataset.ReadAsArray()



# 加NDVI去除植被区域，计算像素高度和建筑物标签的相关系数
dataset, _, _, _, _ = tools.open_tif(realllabel_path)
real_arr = dataset.ReadAsArray()
dataset, _, _, _, _ = tools.open_tif(ndvi_path)
ndvi = dataset.ReadAsArray()

jie_ndsm[ndvi > 0.2] = 0
s_ndsm[ndvi > 0.2] = 0

label = (real_arr[0, :, :] == 0) & (real_arr[1, :, :] == 0) & (real_arr[2, :, :] == 255)

label = label.ravel()
s_ndsm = s_ndsm.ravel()
jie_ndsm = jie_ndsm.ravel()


label_mean = np.mean(label)
label_std = np.std(label)
s_mean = np.mean(s_ndsm)
s_std = np.std(s_ndsm)
jie_mean = np.mean(jie_ndsm)
jie_std = np.std(jie_ndsm)

print("check:", np.shape(label), np.shape(s_ndsm), np.shape(jie_ndsm))

n = np.shape(label)[0]

s_cov = np.sum((s_ndsm - s_mean) * (label - label_mean)) / s_std / label_std / n
jie_cov = np.sum((jie_ndsm - jie_mean) * (label - label_mean)) / jie_std / label_std / n

print('s_p:', s_cov)
print('jie_p:', jie_cov)





