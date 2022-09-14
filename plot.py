# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np







# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

name_list = ('Siamese Network', 'RF', 'KNN', 'DT', 'SVM')
acc = [0.9825436408977556, 0.9398280802292264, 0.6790830945558739, 0.8739255014326648, 0.5530085959885387]

plt.figure(figsize=(9, 5.562))
plt.ylim(0.5, 1.05)
plt.ylabel('Accuracy')
plt.bar(name_list, acc, width=0.6, color='dodgerblue')

plt.show()
