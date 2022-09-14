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
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)



def printParameters():
    # 打印参数
    print("环境参数: ")
    print("labelAmount: ", labelAmount)
    print("featureLength: ", featureLength)

    print("网络参数: ")
    print("trainBatchSize: ", trainBatchSize)
    print("firstLayOutChannel: ", firstLayOutChannel)
    print("secondLayOutChannel: ", secondLayOutChannel)
    print("linearInput: ", linearInput)
    print("kernelSize: ", kernelSize)
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
    instanceLabel = []                                                      # 每个样本在原始文件中的下标

    i = 0
    while i < input_Length:
        input_row = inputFile.values[i]
        input_feature.append(input_row[:-1].reshape(1, -1))                 # 转换成1行
        input_label.append(input_row[-1])                                   # 标签
        instanceLabel.append(i)
        i += 1

    return input_feature, input_label, instanceLabel






def splitExistedData(Input, Input_label, Index, isSplit, TestSize):
    if isSplit == 1:
        train, test, train_label, test_label, trainIndex, testIndex = train_test_split(
                                  Input, Input_label, Index, test_size=TestSize,
                                  random_state=random.randint(1, 1000),
                                  stratify=Input_label)
        print("train length and test length:", len(train), "  ", len(test))
        return train, test, train_label, test_label, trainIndex, testIndex
    if isSplit == 0:
        return Input, [], Input_label, []


class Train_MatchingRule(Dataset):
    def __init__(self, train_Data, train_Label, idxInstance):    # 实现testData对trainData每个样本都进行匹配
        tempData, tempLabel, tempIsSameLabel = [], [], []
        for i in range(len(train_Data)):
            j = i
            while j < len(train_Data):
                tempData.append(train_Data[i])
                tempLabel.append(train_Data[j])                 # train_Data[i]映射到train_Data[j]
                if train_Label[i] == train_Label[j]:
                    tempIsSameLabel.append(1)
                else:
                    tempIsSameLabel.append(0)
                j += 1
        print("Train DataSet Length: ", len(tempData))

        data_root_tensor = torch.FloatTensor(np.array(tempData).tolist())
        data_label_tensor = torch.FloatTensor(np.array(tempLabel).tolist())

        self.data = data_root_tensor
        self.label = data_label_tensor
        self.isSameLabel = tempIsSameLabel

    def __getitem__(self, idx):
        datas = self.data[idx]
        labels = self.label[idx]
        isSameLabel = self.isSameLabel[idx]
        return datas, labels, isSameLabel, 0, 0

    def __len__(self):
        return len(self.data)



class Test_MatchingRule(Dataset):
    def __init__(self, test_Data, train_Data, test_Label, train_Label):    # 全排列规则匹配
        tempData, tempLabel, tempIsSameLabel = [], [], []
        for i in range(len(test_Data)):
            for j in range(len(train_Data)):
                tempData.append(test_Data[i])
                tempLabel.append(train_Data[j])
                if test_Label[i] == train_Label[j]:
                    tempIsSameLabel.append(1)
                else:
                    tempIsSameLabel.append(0)
        print("Test DataSet Length: ", len(tempData))

        data_root_tensor = torch.FloatTensor(np.array(tempData).tolist())
        data_label_tensor = torch.FloatTensor(np.array(tempLabel).tolist())

        self.data = data_root_tensor
        self.label = data_label_tensor
        self.isSameLabel = tempIsSameLabel

    def __getitem__(self, idx):
        datas = self.data[idx]
        labels = self.label[idx]
        isSameLabel = self.isSameLabel[idx]
        return datas, labels, isSameLabel

    def __len__(self):
        return len(self.data)



def createLoader(input_feature, input_label, idx, isSplit, test_Size):
    """
    创建训练集和数据集的 Dataloader
    isSplit==1说明划分了训练集和数据集, 两个都要 DataLoader; ==0说明是跨场景测试, 无需测试的 Loader
    trainData是源场景 train 训练集数据, testData 是目标场景 test 训练集数据. loader对应数据集
    没有使用到source的测试集, 只用source的训练集进行样本对匹配
    :param input_feature:
    :param input_label:
    :param idx:
    :param isSplit:
    :param test_Size:
    :return:
    '"""
    train, test, train_label, test_label, trainIndex, testIndex = \
        splitExistedData(input_feature, input_label, idx, isSplit, test_Size)


    '''
    # 识别新设备, 假定某设备A不参与孪生网络的训练, 但在测试集中出现
    train_temp, train_label_temp = [], []
    for sample_index in range(len(train)):
        if train_label[sample_index] != 13:                             # 编号为 13 的设备记为新设备
            train_temp.append(train[sample_index])
            train_label_temp.append(train_label[sample_index])
    train = train_temp
    train_label = train_label_temp
    '''

    trainData = Train_MatchingRule(train, train_label, trainIndex)
    train_loader = DataLoader(trainData, batch_size=trainBatchSize, shuffle=True)

    if isSplit == 1:                                                               # 有划分才用测试集, 无划分一律当训练集
        testData = Test_MatchingRule(test, train, test_label, train_label)
        test_loader = DataLoader(testData, batch_size=len(train), shuffle=False)
    else:                                                                          # 无划分，不需要 test_loader
        test_loader = []
    return train_loader, test_loader, train_label, test_label






# 自定义 ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.07):
        print("margin: ", margin)
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, Output1, Output2, Label):
        Euclidean_Distance = F.pairwise_distance(Output1, Output2, keepdim=True)

        # 如果是不同类, 损失函数就是两个样本的欧氏距离; 如果同类, 损失函数就是margin-欧氏距离
        # 应该是要希望不同类距离大, 同类距离小, 因此margin可能需要够小才行

        loss_contrastive = torch.mean((Label.float()) * torch.pow(Euclidean_Distance.squeeze(-1), 2) +
                                      (1-Label.float()) * torch.pow(
            torch.clamp(self.margin - Euclidean_Distance.squeeze(-1), min=0.0), 2))
        return loss_contrastive


def initialModel(FirstLayOutChannel, SecondLayOut_Channel, LinearInput, KernelSize, FeatureLength, LearningRate):
    """
    初始化孪生网络模型
    :param FirstLayOutChannel: cnn第一层通道
    :param SecondLayOut_Channel: cnn第二层通道
    :param LinearInput: cnn后Ann的输入长度
    :param KernelSize: 一维卷积核长度
    :param FeatureLength:
    :param LearningRate: 学习率
    :return:
    """
    Model = NetworkModel.SiameseNetworkCNN(FirstLayOutChannel, SecondLayOut_Channel, LinearInput, KernelSize, FeatureLength)
    Criterion = ContrastiveLoss()                               # 定义损失函数
    Optimizer = optim.Adam(Model.parameters(), LearningRate)    # 定义优化器
    if isGPUAvailable:
        return Model.cuda(), Criterion, Optimizer
    else:
        return Model, Criterion, Optimizer











if __name__ == '__main__':
    isGPUAvailable = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("whether GPU is available:", isGPUAvailable)

    # 环境参数
    labelAmount = 15                    # 设备类型数量
    featureLength = 2000

    trainBatchSize = 256
    firstLayOutChannel = 24
    secondLayOutChannel = 32
    linearInput = 1024
    kernelSize = 200                    # 一个ReActor滑动窗口长度
    learningRate = 0.001

    testSize = 0.75
    trainEpoch = 100
    repeatTime = 7
    if isGPUAvailable:
        torch.cuda.set_device(0)

    printParameters()


    inputFileName = "dataOrigin.csv"
    input, inputLabel, index = readAllData(inputFileName)



    # ---------------------多次遍历测试----------------------
    accList = []
    for repeat in range(repeatTime):
        print("--------------------------------------------------------------------------")
        print("repeat time ", (repeat+1))
        epoch_list = []                             # 保存epoch
        trainLoss_List = []                         # 保存每个epoch对应的loss值
        testLoss_List = []
        drawAccList = []

        trainLoader, testLoader, trainLabel, targetLabel = createLoader(input, inputLabel, index, isSplit=1, test_Size=testSize)
        model, criterion, optimizer = \
            initialModel(firstLayOutChannel, secondLayOutChannel, linearInput, kernelSize, featureLength, learningRate)




        # 开始训练
        bestEpochResult = 0
        bestEpochPosition = -1
        for epoch in range(trainEpoch):

            train_startTime = time.time()

            trainLossTemp = 0.0
            epoch_list.append(epoch)
            model.train()
            for i, data in enumerate(trainLoader, 0):
                img0, img1, label, img1Index, img2Index = data
                if isGPUAvailable:
                    img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                optimizer.zero_grad()
                output1, output2 = model(img0, img1)
                Trainloss_contrastive = criterion(output1, output2, label)
                trainLossTemp += Trainloss_contrastive
                Trainloss_contrastive.backward()
                optimizer.step()
            trainLoss_List.append(Trainloss_contrastive)

            train_endTime = time.time()
            print("单次训练所需的时间 = ", (train_endTime-train_startTime), "s.")

            model.eval()                                    # 测试模式, 模型参数不更新
            total = 0                                       # 测试集多少个样本
            correct = 0                                     # 测试集预测正确了几个样本
            testLabelCur = 0                                # 测试集标签的浮标
            testLossAmount = 0                              # 测试集总损失值

            Error_samples = [0] * labelAmount               # 记录每个类别的错误数量
            count = [0] * labelAmount
            avgError_samples = [0] * labelAmount            # 记录每个类别的错误占比 = Error_samples / count

            actualList, predictList = [], []


            for i, data in enumerate(testLoader, 0):
                img0, img1, label = data
                if isGPUAvailable:
                    img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()



                output1, output2 = model(img0, img1)
                testloss_contrastive = criterion(output1, output2, label)
                testLossAmount += testloss_contrastive.item()



                # print("输入前 : ", img0.equal(img1), "   输出后 : ", output1.equal(output2))


                # 一次读取的batchSize就是一个测试样本对于所有训练样本的距离, 对每个类别求距离和, 然后将测试样本归类到距离最短的一类

                distance = [0] * labelAmount                                    # 记录到每个类别的距离之和
                number = [0] * labelAmount                                      # 训练样本中每个类别的总样本数
                avg_distance = [0] * labelAmount                                # 记录到每个类别的平均距离 = distance / number

                euclidean_distance = F.pairwise_distance(output1, output2)      # 计算欧氏距离
                for j in range(len(euclidean_distance)):
                    number[int(trainLabel[j])] += 1
                    distance[int(trainLabel[j])] += euclidean_distance[j].item()

                for k in range(labelAmount):
                    # if k == 13:
                        # continue
                    avg_distance[k] = distance[k] * 1.0 / number[k]

                # avg_distance.pop(13)

                predicted = avg_distance.index(min(avg_distance))

                # if targetLabel[testLabelCur] == 13:
                #     print(min(avg_distance), ", 13")
                # else:
                #     print(min(avg_distance))

                # print('实际类别 : ' + str(targetLabel[testLabelCur]) + ', 预测类别 : ' + str(predicted) + '.')

                # if min(avg_distance) > 0.07:
                    # 如果对于训练集中所有类的平均距离的最小值仍大于阈值, 则该测试样本属于新设备
                    # predicted = 13

                actualList.append(targetLabel[testLabelCur])
                predictList.append(predicted)

                if predicted == targetLabel[testLabelCur]:
                    total += 1
                else:
                    Error_samples[int(targetLabel[testLabelCur])] += 1

                count[int(targetLabel[testLabelCur])] += 1
                testLabelCur += 1

            for k in range(labelAmount):
                avgError_samples[k] = 1.0 - Error_samples[k] * 1.0 / count[k]

            testLoss_List.append(testLossAmount)


            # print("avgError_samples : ", avgError_samples, "\n")                          # 每种设备的错误比例
            # print("confusion_matrix : ", confusion_matrix(actualList, predictList), "\n")    # 混淆矩阵

            print("epoch: ", epoch + 1, "  trainLoss: ", Trainloss_contrastive.item(),
                  "  testLoss: ", testloss_contrastive.item(), "  Accuracy: ", total / len(targetLabel),
                  "  testLossAmount: ", testLossAmount, "\n")

            correct = total / len(targetLabel)              # 张量之间的比较运算
            drawAccList.append(correct)

            if correct > bestEpochResult:
                bestEpochResult = correct
                bestEpochPosition = epoch + 1

        accList.append(bestEpochResult)
        print("当前epoch中最好的准确率: ", bestEpochResult)
