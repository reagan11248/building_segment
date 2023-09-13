# coding:utf-8

import numpy as np
import sklearn.model_selection as sk_model_selection
import time
#import LSDtest
import matplotlib.pyplot as plt
import cv2
import os
import math
import random
import tools
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('error', RuntimeWarning)

def printtime(t):
    t.append(time.time())
    cost = t[-1] - t[-2]
    s = cost % 60
    m = (cost - s) % 3600 // 60
    h = cost // 3600
    print('耗时：%d:%d:%d' % (h, m, s))



def feature_read(index_path, index_names):
    X = None
    y = None

    file_list = os.listdir(index_path)
    for file in file_list:
        img_f = None
        for index_name in index_names:

            tif_path = index_path + '/' + file + '/' + index_name + '.tif'
            index_dataset, cols, rows, geotransform, space = open_tif(tif_path)
            index_array = index_dataset.ReadAsArray()
            index_array = index_array.flatten()

            if img_f is None:
                img_f = index_array
            else:
                img_f = np.hstack((img_f, index_array))
        img_f = np.transpose(img_f, [1, 0])

        if X is None:
            X = img_f
        else:
            X = np.hstack((X, img_f))


    for file in file_list:
        tif_path = index_path + '/' + file + '/decision_tree_label.tif'
        index_dataset, cols, rows, geotransform, space = open_tif(tif_path)
        y1 = index_dataset.ReadAsArray()
        y1 = y1.flatten()

        if y is None:
            y = y1
        else:
            X = np.hstack((X, img_f))



    return




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

    #print('table_number:', np.dtype(table_number[0]))
    #print(table_number)
    #preprocessing.scale(X_rgb, axis=0, copy=False)
    # 标准化
    print('标准化')
    print(np.dtype(X[0, 0]))
    print(np.shape(X))
    #pca = PCA(n_components=0.99)
    '''
    #scaler = preprocessing.scale()

    X_rgb = pca.fit_transform(X[:, 0:768])
    X_rgb = preprocessing.scale(X_rgb)
    print('X_rgb:', np.shape(X_rgb))

    X_gaber = preprocessing.scale(X[:, 768:780])

    X_lbp = pca.fit_transform(X[:, 780:888])
    #X_lbp = preprocessing.scale(X_lbp)

    X_slop = preprocessing.scale(X[:, 888:920])

    X_dsm = preprocessing.scale(X[:, 920:922])

    X_rgb_ms = X[:, 922:928]

    X = np.hstack((X_rgb, X_gaber, X_lbp, X_rgb_ms))

    #, X_slop, X_dsm
    #X = preprocessing.scale(X, axis=0, copy=False)

    #scale = preprocessing.MinMaxScaler(copy=False)
    #scale.fit_transform(X)

    '''
    #X = pca.fit_transform(X)

    X = preprocessing.scale(X)

    print(np.shape(X)[1])

    # 方差选择法，返回值为特征选择后的数据
    # 参数threshold为方差的阈值
    #VarianceThreshold(threshold=3).fit_transform(data.data)


    N = np.shape(X)[1]

    # pu-learing
    NP = np.shape(X[y == 1])[0]
    NU = np.shape(X[y == 0])[0]
    print('正样本数：%d' % NP)
    print('负样本数：%d' % NU)

    #############################################################
    #cross_validation(NU, NP)

    ##############################################################
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
    # pool.map(trainProcess, [[NU, NP] for _ in range(ite)])

    pool.close()
    pool.join()

    pre_proba_arr = np.frombuffer(pre_proba.get_obj(), dtype=np.float64)
    pre_n_arr = np.frombuffer(pre_n.get_obj(), dtype=np.int)

    pre_label = pre_proba_arr / pre_n_arr
    ########################################################
    #np.savez(label_path + '\\pre_proba_arr.npz', pre_proba_arr)
    t_path = os.path.abspath(os.path.join(table_path, ".."))
    np.savez(t_path + '\\pre_label.npz', pre_label)
    ##################################################
    y = y.astype(np.float64)
    y[y == 0] = pre_label

    for file_index, filename in enumerate(tif_list):
        if file_index == 0:
            start = 0
        else:
            start = table_number[:file_index].sum()
        end = start + table_number[file_index]
        np.savez(table_path + '/' + filename + '/' + 'result.npz', y[start:end])







if __name__ == '__main__':
    path = os.path.abspath('..')
    img_path = path + '/org/img'
    reallable_path = path + '/org/reallable'
    segment_path = path + '/org/segment'
    dsm_path = path + '/org/dsm'

    index_path = path + '/temporary/index'
    label_path = path + '/temporary/label'
    table_path = path + '/temporary/table'
    result_path = path + '/temporary/result'
    evaluate_path = path + '/temporary/evaluate'
    d_t_path = path + '/temporary/d_t'
    total_result_path = path + '/temporary/total_result'
    process_path = path + '/temporary/process'
    process2_path = path + '/temporary/process2'

    for p in [index_path, label_path, table_path, result_path, evaluate_path, d_t_path, total_result_path, process_path,
              process2_path]:
        if not os.path.exists(p):
            os.mkdir(p)

    t = [time.time()]















