# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable



class SiameseNetworkCNN(nn.Module):
    def __init__(self, firstLayOutChannel, secondLayOutChannel, linearInput, kernelSize, featureLength):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, firstLayOutChannel, kernelSize),
            nn.BatchNorm1d(firstLayOutChannel),         # batchNorm1d's input is the outchannel of conv1d
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(firstLayOutChannel, secondLayOutChannel, kernelSize),
            nn.BatchNorm1d(secondLayOutChannel),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            torch.nn.Linear((secondLayOutChannel * (featureLength - 2 * kernelSize + 2)) // 2, linearInput),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(linearInput, linearInput // 2),
            torch.nn.Linear(linearInput // 2, 256))

    def forward_once(self, x):
        out = self.layer1(x)
        out = self.layer2(out)                          # output也是batch_first, 实际上h_n与c_n并不是batch_first
        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2






#4 layers ANN network
class ANN_4Hidden_Net(nn.Module):
    def __init__(self,in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(ANN_4Hidden_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.layer5(x)
        return out
