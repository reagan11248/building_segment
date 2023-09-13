#coding:utf-8

import numpy as np
import sklearn.model_selection as sk_model_selection
import time
import matplotlib.pyplot as plt
import cv2
import os
import math
import random
import tools
from sklearn import metrics
import matplotlib.pyplot as plt
import shutil
import warnings
warnings.simplefilter('error', RuntimeWarning)


'''
accs=sk_model_selection.cross_val_score(model, iris_X, y=iris_y, scoring='f1', cv=10, n_jobs=1)
print('交叉验证结果:', accs)
'''


def printtime(t):
    t.append(time.time())
    cost = t[-1] - t[-2]
    s = cost % 60
    m = (cost - s) % 3600 // 60
    h = cost // 3600
    print('耗时：%d:%d:%d' % (h, m, s))



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

    ################################################################

    print('计算图像特征...')
    
    tools.normaliz(img_path, index_path, max_value=255)
    
    #tools.rgb_i(img_path, index_path)
    
    tools.ndvi(img_path, index_path)
    
    tools.gabor_image(index_path)

    # lbp

    tools.LBP(index_path)
    

    tools.dsm_slop(dsm_path, index_path)
    
    tools.dsm_process(dsm_path, index_path)
    
    
    # 阴影影响
    degree = 10
    influence_length = 40
    tools.fuzzy_shadow_landscape3(index_path, degree, influence_length)
    printtime(t)


    print('分割...')
    tools.slic_segment_muti(path)
    printtime(t)
    ######################################################################
    print('区域特征统计...')

    m_s = ['fuzzy_shadow_landscape', 'blur_dsm', 'r', 'g', 'b']

    for fr in [4, 6]:
        for th in range(1, 9):
            pass
            m_s.append('gabor_fr_' + str(fr) + '_th_' + str(th))
        m_s.append('gabor_fr_' + str(fr) + '_max')


    h_e_slop = [i for i in range(32)]
    h1 = 'lbp1'
    h3 = 'lbp3'
    h5 = 'lbp5'
    h_r = 'r'
    h_g = 'g'
    h_b = 'b'
    h_l1 = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 37, 39, 43, 45, 47, 51, 53, 55, 59, 61, 63,
            85, 87, 91, 95, 111, 119, 127, 255]
    hcolor = [i for i in range(256)]

    h = [[h1, h_l1], [h3, h_l1], [h5, h_l1], ['e_slop', h_e_slop]]

    tif_list = os.listdir(path + '/org/img')
    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            out_path = table_path + '/' + os.path.splitext(filename)[0]
            if not os.path.exists(out_path):
                os.mkdir(out_path)

            segment_img = label_path + '/' + filename

            for i in m_s:
                index_img = index_path + '/' + os.path.splitext(filename)[0] + '/' + i + '.tif'
                tools.block_mea_std(index_img, segment_img, out_path)
                print(i, 'm_s')
            
            for i in h:
                index_img = index_path + '/' + os.path.splitext(filename)[0] + '/' + i[0] + '.tif'
                tools.block_histogram(index_img, segment_img, i[1], out_path)
                print('h')
            
            print('统计完成：', filename)


    # gabor
    tools.block_gabor(table_path)

    printtime(t)

    ##################################################################
    print('决策树分类...')
    tif_list = os.listdir(path + '/org/img')
    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            data_path = table_path + '/' + os.path.splitext(filename)[0]
            tools.decision_tree(data_path)

    printtime(t)

    ##################################################################
    print('决策树分类评价...')

    th_list = np.array([0.5, 0.6])
    n = 0
    tpr = np.zeros_like(th_list)
    fpr = np.zeros_like(th_list)
    m = np.zeros_like(th_list)
    p = np.zeros_like(th_list)
    r = np.zeros_like(th_list)
    f1 = np.zeros_like(th_list)

    tif_list = os.listdir(path + '/org/img')
    data = np.zeros((4, np.shape(th_list)[0]))
    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            data_path = table_path + '/' + os.path.splitext(filename)[0] + '/decision_tree_label.npz'
            segment_path = label_path + '/' + filename
            out_path = d_t_path + '/' + os.path.splitext(filename)[0]
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            # lable转图
            tools.data2tif(data_path, segment_path, out_path + '/decision_tree.tif')
            # 计算准确度
            real_label_path = path + '/org/reallable/' + filename
            evaluat_path = out_path + '/evaluation'
            if not os.path.exists(evaluat_path):
                os.mkdir(evaluat_path)
            data += tools.evaluation(th_list, real_label_path, out_path + '/decision_tree.tif', evaluat_path)

    # 总
    out_path = d_t_path + '/total'
    data_path = out_path + '/data.txt'
    evaluate_path = out_path

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(data_path, 'w')

    true_positive, true_negative, false_positive, false_negative = data

    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (false_positive + true_negative)
    auc = metrics.auc(tpr, fpr)
    fig = plt.figure()
    x = fpr + [1]
    y = tpr + [1]

    plt.plot(x, y, label='ROC')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC\nAUC = " + str(auc))
    plt.legend()

    plt.savefig(evaluate_path + '\\roc.png')
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

    plt.savefig(evaluate_path + '\\MIoU.png')
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

    plt.savefig(evaluate_path + '\\precision&recall.png')
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

    plt.savefig(evaluate_path + '\\f1.png')
    plt.cla()

    plt.close()

    f.write('f1\n')
    f.write('th_list,f1\n')
    for i in range(len(th_list)):
        f.write(str(th_list[i]) + ',' + str(f1[i]) + '\n')

    f.close()

    printtime(t)


    print('训练分类器...')
    
    f_name = ['r_histogram', 'g_histogram', 'b_histogram',
              'gabor_fr_4_mean_std', 'gabor_fr_6_mean_std',
              'gabor_fr_4_max_mean_std', 'gabor_fr_6_max_mean_std',
              'lbp1_histogram', 'lbp3_histogram', 'lbp5_histogram',
              'e_slop_histogram', 'blur_dsm_mean_std',
              'r_mean_std', 'g_mean_std', 'b_mean_std'
              ]

    ite = 100
    Parallel_number = 10

    tools.pu_learing1(table_path, f_name, ite, Parallel_number)

    #tools.train_OCSVM_rbf(table_path, f_name)

    #tools.train_OCSVM_poly(table_path, f_name)

    #tools.train_OCSVM_sig(table_path, f_name)

    #tools.train_Iso(table_path, f_name)

    #tools.train_KDE(table_path, f_name)

    #tools.train_S_EM(table_path, f_name, 15)

    printtime(t)

    ##################################################################

    print('结果评价...')
    th_list = np.arange(0, 1, 0.01)
    n = 0
    tpr = np.zeros_like(th_list)
    fpr = np.zeros_like(th_list)
    m = np.zeros_like(th_list)
    p = np.zeros_like(th_list)
    r = np.zeros_like(th_list)
    f1 = np.zeros_like(th_list)

    tif_list = os.listdir(path + '/org/img')
    data = np.zeros((4, np.shape(th_list)[0]))
    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            data_path = table_path + '/' + os.path.splitext(filename)[0] + '/result.npz'
            segment_path = label_path + '/' + filename
            out_path = result_path + '/' + os.path.splitext(filename)[0]
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            # lable转图
            tools.data2tif(data_path, segment_path, out_path + '/result.tif')
            # 计算准确度
            real_label_path = path + '/org/reallable/' + filename
            evaluat_path = out_path + '/evaluation'
            if not os.path.exists(evaluat_path):
                os.mkdir(evaluat_path)
            data += tools.evaluation(th_list, real_label_path, out_path + '/result.tif', evaluat_path)

    # 总
    out_path = result_path + '/total'
    data_path = out_path + '/data.txt'
    evaluate_path = out_path

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(data_path, 'w')
    
    true_positive, true_negative, false_positive, false_negative = data
    
    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (false_positive + true_negative)
    auc = metrics.auc(fpr, tpr)
    fig = plt.figure()
    x = fpr + [1]
    y = tpr + [1]

    plt.plot(x, y, label='ROC')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC\nAUC = " + str(auc))
    plt.legend()

    plt.savefig(evaluate_path + '\\roc.png')
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

    plt.savefig(evaluate_path + '\\MIoU.png')
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

    plt.savefig(evaluate_path + '\\precision&recall.png')
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

    plt.savefig(evaluate_path + '\\f1.png')
    plt.cla()

    plt.close()

    f.write('f1\n')
    f.write('th_list,f1\n')
    for i in range(len(th_list)):
        f.write(str(th_list[i]) + ',' + str(f1[i]) + '\n')

    f.close()

    printtime(t)

    ##################################################################
    print('后处理...')
                                                # 【二值化、填充空洞、小区域删除、开闭操作】
    thr_min, thr_max = 0.2, 0.7

    tools.muti_processing1(result_path, img_path, label_path, process2_path, table_path, index_path, thr_min, thr_max)
    printtime(t)

    ############################################################
    print('后处理评价...')

    th_list = np.arange(0, 255, 128)
    tif_list = os.listdir(path + '/org/img')
    data = np.zeros((4, np.shape(th_list)[0]))
    for filename in tif_list:
        if os.path.splitext(filename)[1] == '.tif':
            real_label_path = reallable_path + '/' + filename
            pre_path = process2_path + '/' + os.path.splitext(filename)[0] +'/binarization.tif'
            out_path = process2_path + '/' + os.path.splitext(filename)[0] +'/evaluation'
            if not os.path.exists(process2_path + '/' + os.path.splitext(filename)[0]):
                os.mkdir(process2_path + '/' + os.path.splitext(filename)[0])
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            data += tools.evaluation(th_list, reallable_path + '/' + filename, process2_path + '/' +  os.path.splitext(filename)[0] + '/binarization.tif', out_path)
            #print(pre_path)
    
    data_path = process2_path + '/data.txt'
    evaluate_path = process2_path + '/evaluate'

    if not os.path.exists(evaluate_path):
        os.mkdir(evaluate_path)

    f = open(data_path, 'w')

    true_positive, true_negative, false_positive, false_negative = data

    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (false_positive + true_negative)
    auc = metrics.auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC\nAUC = " + str(auc))
    plt.legend()

    plt.savefig(evaluate_path + '\\roc.png')
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

    plt.savefig(evaluate_path + '\\MIoU.png')
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

    plt.savefig(evaluate_path + '\\precision&recall.png')
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

    plt.savefig(evaluate_path + '\\f1.png')
    plt.cla()

    plt.close()

    f.write('f1\n')
    f.write('th_list,f1\n')
    for i in range(len(th_list)):
        f.write(str(th_list[i]) + ',' + str(f1[i]) + '\n')

    f.close()



