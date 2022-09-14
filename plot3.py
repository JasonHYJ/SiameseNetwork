# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_twin(_y1, _y2, _y3, _y4, _ylabel1, _ylabel2):
    x = ['500', '800', '1000', '1500', '2000']
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('输入特征长度')
    ax1.set_ylabel(_ylabel1, color=color)
    l1, = ax1.plot(x, _y1, color=color, label="孪生网络")
    l3, = ax1.plot(x, _y3, color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.set(ylim=[0.7, 1.2])


    for a, b in zip(x, _y1):
        plt.text(a, b+0.015, '%.4f' % b, ha='center', va= 'bottom',fontsize=9)

    for a, b in zip(x, _y3):
        plt.text(a, b-0.02, '%.4f' % b, ha='center', va= 'bottom',fontsize=9)


    ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴

    color = 'tab:red'
    ax2.set_ylabel(_ylabel2, color=color)
    l2, = ax2.plot(x, _y2, color=color, label="传统机器学习")
    l4, = ax2.plot(x, _y4, color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    ax2.set(ylim=[-20, 350])

    for a, b in zip(x, _y2):
        plt.text(a, b+0.035, '%.0f' % b, ha='center', va= 'bottom',fontsize=9)

    for a, b in zip(x, _y4):
        plt.text(a, b+0.035, '%.2f' % b, ha='center', va= 'bottom',fontsize=9)

    ax2.legend([l1, l2, l3, l4], ['孪生网络', '传统机器学习', '孪生网络', '传统机器学习'])

    fig.tight_layout()
    plt.show()


def Plot(_y1, _y2):
    x = ['500', '800', '1000', '1500', '2000']
    plt.plot(x, acc, label="孪生网络")
    plt.plot(x, acc_rf, linestyle='--', label="传统机器学习(KNN)")
    plt.ylim(0.9, 1.0)

    for a, b in zip(x, acc):
        plt.text(a, b+0.007, '%.4f' % b, ha='center', va= 'bottom',fontsize=9)

    for a, b in zip(x, acc_rf):
        plt.text(a, b-0.008, '%.4f' % b, ha='center', va= 'bottom',fontsize=9)
    plt.xlabel("输入特征长度")
    plt.ylabel("识别准确率")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    # 创建模拟数据
    featureLength = ['500', '800', '1000', '1500', '2000']
    x = range(len(featureLength))
    acc = [0.962568578553616, 0.9650872817955112, 0.9825436408977556, 0.9451371571072319, 0.9301745635910225]
    time_cost = [0.45400404930114746 * 100, 0.8635389804840088 * 100, 1.2192676067352295 * 100, 2.072669506072998 * 100, 2.923171281814575 * 100]

    acc_rf = [0.9451122194513716, 0.9451371571072319, 0.9376558603491272, 0.9276807980049875, 0.932643391521197]
    time_cost_tf = [0.21645593643188477, 0.24637627601623535, 0.2759835720062256, 0.3732619285583496, 0.4354987144470215]

    plot_twin(acc, time_cost, acc_rf, time_cost_tf, '识别准确率', '训练时间开销（秒）')

    Plot(acc, acc_rf)