# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


plt.figure(figsize=(9, 5.562))
name_list = ('Siamese Network', 'RF', 'KNN', 'DT', 'SVM')
acc = [0.9825436408977556, 0.9398280802292264, 0.6790830945558739, 0.8739255014326648, 0.5530085959885387]
acc_new = [0.9276807980049875, 0.8681948424068768, 0.6389684813753582, 0.8080229226361032, 0.508567335243553]

bar_width = 0.3  # 条形宽度
index_entropy = np.arange(len(name_list))
index_IoTArogs = index_entropy + bar_width

# 使用两次 bar 函数画出两组条形图
plt.bar(index_entropy, height=acc, width=bar_width, color='coral', label='类型识别')
plt.bar(index_IoTArogs, height=acc_new, width=bar_width, color='dodgerblue', label='新设备检测')

plt.legend(fontsize=13)  # 显示图例
plt.xticks(index_entropy + bar_width/2, name_list, fontsize=13)
plt.yticks(fontsize=13)


# plt.show()
plt.savefig('fig3.pdf')
