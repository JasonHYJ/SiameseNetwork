# -*- coding: utf-8 -*-

import itertools
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    sns.set()         # 设置字体大小
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', verticalalignment='center', color='white'
        if cm[i, j] > thresh else 'black')
        # 方格内的文字水平居中, 竖直居中
    plt.tight_layout()
    # plt.ylabel('True label', fontsize=15)
    # plt.xlabel('Predicted label', fontsize=15)

    plt.tight_layout()          # 设置横纵坐标标签自适应, 不重叠
    plt.savefig('confusion matrix.pdf')
    # plt.show()



if __name__ == "__main__":
    cnf_matrix = np.array([[4, 2, 0, 0, 2, 6, 1],
                           [0, 100, 3, 1, 1, 11, 0],
                           [0, 2, 73, 0, 0, 2, 0],
                           [0, 0, 0, 51, 0, 6, 0],
                           [1, 0, 0, 0, 383, 4, 0],
                           [1, 2, 1, 2, 0, 991, 0],
                           [1, 0, 0, 0, 1, 4, 54]])

    cnf_matrix = np.array([[4, 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
                           [0, 34, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                           [0, 0,  28, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                           [0, 0,  0,  30, 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
                           [0, 0,  0,  0,  29, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                           [0, 0,  0,  0,  0,  29, 0,  0,  0,  0,  0,  0,  0,  0,  0],
                           [0, 0,  0,  0,  0,  0,  31, 0,  0,  0,  0,  0,  0,  0,  0],
                           [0, 0,  0,  0,  0,  0,  0,  26, 0,  0,  0,  0,  0,  0,  0],
                           [0, 0,  0,  0,  0,  0,  0,  0,  36, 0,  0,  0,  0,  0,  0],
                           [0, 0,  0,  0,  0,  0,  0,  0,  0,  24, 0,  0,  0,  0,  0],
                           [0, 0,  1,  0,  0,  0,  0,  0,  0,  0,  28, 0,  0,  0,  0],
                           [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  28, 0,  0,  0],
                           [0, 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  31, 0,  0],
                           [0, 0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  26, 0],
                           [0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10]])

    attack_types = ["Smart_Things", "Amazon_Echo", "Netatmo_Welcome", "TP-Link_Day_Night_Cloud_camera", "Samsung_SmartCam",
                    "Withings_Smart_Baby_Monitor", "Belkin_wemo_switch", "TP-Link_Smart_plug", "Belkin_wemo_motion_sensor",
                    "NEST_Protect_smoke_alarm", "Netatmo_weather_station", "Withings_Aura_smart_sleep_sensor", "LiFX_Bulb",
                    "PIX-STAR_Photo-frame", "Nest_Dropcam"]

    plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True)