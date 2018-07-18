"""
model

C3D input size = [1,3,48,256,256] 
    48f씩 sliding window 256*256 size


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class C3D(nn.Module):
    """
    nb_classes: nb_classes in classification task, 101 for UCF101 dataset
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


"""
Neural Networks model : GRU
"""

# simple GRU. non-bidirectional
class GRU(nn.Module):

    def __init__(self, c3d):
        super(GRU, self).__init__()

        #C3D를 통과한 snippet을 GRU cell로.
        self.c3d = c3d

        # input_size : input x에서 뽑아낸 expected features의 수
        # hidden_size : hidden state h의 feature 수
        # input_size = 243, hidden_size = 20

        self.gru = nn.GRUCell(243,10).cuda()


    def forward(self, input):

        start = 0
        end = 48
        #(batch=128, hidden=1) tensor create
        hidden_total = torch.FloatTensor(128,10).normal_().cuda()

        input = input.permute(0,2,1,3,4)
        temporal_pool = nn.MaxPool1d(4, 4, 0)

        step = 0
        while end < input.shape[2]:
            x = input[:, :, start:end, :, :]  # x.shape: 1, 3, 48, h, w
            hidden = self.c3d(x)  # c3d forwarding => 1, 512, 3, 9, 9
            hidden = hidden.squeeze()
            hidden = hidden.view(1, 512, -1).permute(0, 2, 1)
            hidden = temporal_pool(hidden).permute(0, 2, 1).squeeze()

            hidden_total = (self.gru(hidden.cuda(), hidden_total))
            print("snippet:", step, hidden_total)

            start += 6
            end += 6
            step += 1

        print(len(hidden_total))  # 128

        return hidden_total
