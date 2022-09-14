# -*- coding: utf-8 -*-

import random as rd
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


def Read_csv(trainFile, testFile):
     # 读取整合的csv文件，并且开始做分类处理
     train_data = pd.read_csv(trainFile)
     test_data = pd.read_csv(testFile)
     train_dataLen, train_dataWid = train_data.shape
     test_dataLen, test_dataWid = test_data.shape

     train = []
     test=[]
     train_label = []
     test_label = []
     # 读取数据
     for i in range(train_dataLen):
          train_row = train_data.values[i]
          train.append(train_row[0 : train_dataWid - 1])
          train_label.append(train_row[-1])
     for i in range(test_dataLen):
          test_row = test_data.values[i]
          test.append(test_row[0 : test_dataWid - 1])
          test_label.append(test_row[-1])

     return train, train_label, test, test_label






def readAllData(InputFileName):
    """
    :param InputFileName: 输入数据的csv文件名
    :return input: 每一个样本的特征大小都是 1 * featureLength
    """
    inputFile = pd.read_csv(InputFileName, encoding='utf-8', header=None)
    input_Length, input_dataWidth = inputFile.shape
    # 获得数据矩阵长和宽, 长表示样本数量, 宽表示每个样本的特征维度

    input_feature = []
    input_label = []

    i = 0
    while i < input_Length:
        input_row = inputFile.values[i]
        input_feature.append(input_row[:-1].reshape(1, -1))
        input_label.append(input_row[-1])                                   # 标签
        i += 1

    train, test, train_label, test_label= train_test_split(
                              input_feature, input_label, test_size=0.75,
                              random_state=rd.randint(1, 1000),
                              stratify=input_label)
    print("train length and test length:", len(train), "  ", len(test))
    return train, test, train_label, test_label





# ========================分类算法=================================================================
def knn(Train_file, Test_file, feature_length):
     train, train_label, test, test_label = Read_csv(Train_file, Test_file)
     # train, test, train_label, test_label = readAllData("dataOrigin.csv")

     # train = np.reshape(train, (-1, feature_length))
     # test = np.reshape(test, (-1, feature_length))

     knn = KNeighborsClassifier()        # KNN 算法
     knn.fit(train, train_label)

     Predict_knn = knn.predict(test).tolist()
     print("KNN算法的准确率 = " + str(knn.score(test, test_label)))
     print(knn.predict_proba(test))
     knn.predict_proba(test)


# ================================================================================================
def rf(Train_file, Test_file, feature_length):
     train, train_label, test, test_label = Read_csv(Train_file, Test_file)
     # train, test, train_label, test_label = readAllData("dataOrigin.csv")
     rf = RandomForestClassifier()      # 随机森林 算法

     # train = np.reshape(train, (-1, feature_length))
     # test = np.reshape(test, (-1, feature_length))

     rf.fit(train, train_label)

     Predict_rf = rf.predict(test).tolist()
     result = rf.score(test, test_label)
     print("随机森林算法的准确率 = " + str(result))

     print(rf.predict_proba(test))
     rf.predict_proba(test)


# ================================================================================================
def dt(Train_file, Test_file, feature_length):
     # train, train_label, test, test_label = Read_csv(Train_file, Test_file)
     train, test, train_label, test_label = readAllData("dataOrigin.csv")

     train = np.reshape(train, (-1, feature_length))
     test = np.reshape(test, (-1, feature_length))


     dt = DecisionTreeClassifier()      # 决策树 算法
     dt.fit(train, train_label)
     Predict_dt = dt.predict(test).tolist()
     result = dt.score(test, test_label)
     print("决策树算法的准确率 = " + str(result))
     dt.predict_proba(test)

# ================================================================================================
def SVM(Train_file, Test_file, feature_length):
     # train, train_label, test, test_label = Read_csv(Train_file, Test_file)
     train, test, train_label, test_label = readAllData("dataOrigin.csv")

     train = np.reshape(train, (-1, feature_length))
     test = np.reshape(test, (-1, feature_length))

     # 调整 SVM 的超参数
     c = list(range(15, 21, 1))
     for i in range(len(c)):
          c[i] = c[i]/10                  # 以 0.1 为间隔的1.5 ~ 2.2的列表
     Gamma = list(range(8,22,1))

     truth_score = [0,0]
     for i in range(len(c)):
          for j in range(len(Gamma)):
               SVM = svm.SVC(C=c[i], kernel='rbf', gamma=Gamma[j], decision_function_shape='ovo')  # 一般c = 2, gamma = 20  支持向量机 算法
               SVM.fit(train, train_label)
               temp = SVM.score(test, test_label)
               if temp > truth_score[1]:
                    truth_score[1] = temp
                    truth_score[0] = SVM.score(train, train_label)

     print("svm的训练集准确率 = " + str(truth_score[0]))
     print("svm的测试集准确率 = " + str(truth_score[1]))
     SVM.predict_proba(test)

# ================================================================================================
def nb(Train_file, Test_file, feature_length):
     # train, train_label, test, test_label = Read_csv(Train_file, Test_file)
     train, test, train_label, test_label = readAllData("dataOrigin.csv")

     train = np.reshape(train, (-1, feature_length))
     test = np.reshape(test, (-1, feature_length))

     nb = GaussianNB()                 # 朴素贝叶斯 算法
     nb.fit(train, train_label)

     Predict_nb = nb.predict(test).tolist()
     result = nb.score(test,test_label)
     print("朴素贝叶斯算法的准确率 = " + str(result))
     nb.predict_proba(test)

# ================================================================================================

if __name__ == "__main__":
     feature_length = 1000
     # start_time = time.time()
     rf("train.csv", "test.csv", feature_length)
     end_time = time.time()
     knn("train.csv", "test.csv", feature_length)
     # dt("train.csv", "test.csv", feature_length)
     # SVM("train.csv", "test.csv", feature_length)
     # nb("train.csv", "test.csv", feature_length)
     # print(end_time - start_time)
