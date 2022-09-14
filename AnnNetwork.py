# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import torch
import time
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import NetworkModel
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)




def printParameters():
    # 打印参数
    print("环境参数: ")
    print("labelAmount: ", labelAmount)

    print("网络参数: ")
    print("BatchSize: ", BatchSize)
    print("linearInput: ", linearInput)
    print("hidden_1: ", hidden_1)
    print("hidden_2: ", hidden_2)
    print("hidden_3: ", hidden_3)
    print("hidden_4: ", hidden_4)
    print("learningRate: ", learningRate)

    print("训练参数: ")
    print("testSize: ", testSize)
    print("trainEpoch: ", trainEpoch)
    print("repeatTime: ", repeatTime)







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

    return input_feature, input_label




def splitExistedData(Input, Input_label, TestSize):
    train, test, train_label, test_label= train_test_split(
                              Input, Input_label, test_size=TestSize,
                              random_state=random.randint(1, 1000),
                              stratify=Input_label)
    print("train length and test length:", len(train), "  ", len(test))
    return train, test, train_label, test_label


def createLoader(input_feature, input_label, test_Size):
    """
    创建训练集和测试集的 Dataloader
    :param input_feature:
    :param input_label:
    :param test_Size:
    :return:
    """
    train, test, train_label, test_label = splitExistedData(input_feature, input_label, test_Size)

    x_train = torch.Tensor(train)
    y_train = torch.Tensor(train_label).type(torch.LongTensor)

    x_test = torch.Tensor(test)
    y_test = torch.Tensor(test_label).type(torch.LongTensor)

    traindata = torch.utils.data.TensorDataset(x_train, y_train)
    testdata = torch.utils.data.TensorDataset(x_test, y_test)


    train_loader = DataLoader(traindata, batch_size=BatchSize, shuffle=True)
    test_loader = DataLoader(testdata, batch_size=BatchSize, shuffle=True)
    return train_loader, test_loader, train_label, test_label





def initialModel(LinearInput, hidden_1, hidden_2, hidden_3, hidden_4, LinearOuput, LearningRate):
    """
    初始化 ANN 模型
    :param LinearInput
    :param hidden_1: ann 第一层节点数
    :param hidden_2: ann 第二层节点数
    :param hidden_3: ann 第三层节点数
    :param hidden_4: ann 第四层节点数
    :param LinearOuput:
    :param LearningRate: 学习率
    :return:
    """
    Model = NetworkModel.ANN_4Hidden_Net(LinearInput, hidden_1, hidden_2, hidden_3, hidden_4, LinearOuput)
    Criterion = nn.CrossEntropyLoss()                               # 定义损失函数
    Optimizer = optim.Adam(Model.parameters(), LearningRate)        # 定义优化器
    if isGPUAvailable:
        return Model.cuda(), Criterion, Optimizer
    else:
        return Model, Criterion, Optimizer






if __name__ == '__main__':
    isGPUAvailable = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("whether GPU is available:", isGPUAvailable)

    # 环境参数
    labelAmount = 15            # 设备类型数量
    linearInput = 1000

    BatchSize = 256
    hidden_1 = 256
    hidden_2 = 512
    hidden_3 = 512
    hidden_4 = 256
    LinearOuput = labelAmount
    learningRate = 0.001

    testSize = 0.75
    trainEpoch = 100
    repeatTime = 7
    if isGPUAvailable:
        torch.cuda.set_device(0)

    printParameters()

    inputFileName = "dataOrigin.csv"
    input, inputLabel = readAllData(inputFileName)

    # ---------------------多次遍历测试----------------------
    accList = []
    for repeat in range(repeatTime):
        print("--------------------------------------------------------------------------")
        print("repeat time ", (repeat + 1))
        epoch_list = []                         # 保存epoch
        trainLoss_List = []                     # 保存每个epoch对应的loss值
        testLoss_List = []
        drawAccList = []

        trainLoader, testLoader, trainLabel, targetLabel = createLoader(input, inputLabel, test_Size=testSize)
        model, criterion, optimizer = \
            initialModel(linearInput, hidden_1, hidden_2, hidden_3, hidden_4, LinearOuput, learningRate)

        # 开始训练
        bestEpochResult = 0
        bestEpochPosition = -1
        for epoch in range(trainEpoch):
            train_startTime = time.time()

            trainLossTemp = 0.0
            epoch_list.append(epoch)
            model.train()

            for i, data in enumerate(trainLoader, 0):
                train, label = data
                if isGPUAvailable:
                    train, label = train.cuda(), label.cuda()
                optimizer.zero_grad()
                output = model(train).squeeze(1)

                print(output)
                print(label)
                print(output.shape)
                print(label.shape)

                Trainloss_CrossEntropy = criterion(output, label.long())
                trainLossTemp += Trainloss_CrossEntropy
                Trainloss_CrossEntropy.backward()
                optimizer.step()
            trainLoss_List.append(Trainloss_CrossEntropy)

            train_endTime = time.time()
            print("单次训练所需的时间 = ", (train_endTime-train_startTime), " s.")



            model.eval()
            total = 0                                       # 测试集多少个样本
            correct = 0                                     # 测试集预测正确了几个样本
            testLossAmount = 0                              # 测试集总损失值

            for j, data in enumerate(testLoader, 0):
                test, label = data
                if isGPUAvailable:
                    test, label = test.cuda(), label.cuda()

                output = model(test)
                testloss_contrastive = criterion(output, label)
                testLossAmount += testloss_contrastive.item()

                testLoss_List.append(testLossAmount)




                print("epoch: ", epoch + 1, "  trainLoss: ", Trainloss_CrossEntropy.item(),
                      "  testLoss: ", testloss_contrastive.item(), "  Accuracy: ", total / len(targetLabel),
                      "  testLossAmount: ", testLossAmount, "\n")

                correct = total / len(targetLabel)
                drawAccList.append(correct)

                if correct > bestEpochResult:
                    bestEpochResult = correct
                    bestEpochPosition = epoch + 1

            accList.append(bestEpochResult)
            print("当前epoch中最好的准确率: ", bestEpochResult)

