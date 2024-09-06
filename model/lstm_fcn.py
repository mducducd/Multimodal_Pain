import torch
from torch import Tensor
import torch.nn as nn
import warnings
from collections import OrderedDict
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')

class LSTM(torch.nn.Module):
    
    def __init__(self, input_length, units, dropout):
    
        '''
        Parameters:
        __________________________________
        input_length: int.
            Length of the time series.

        units: list of int.
            The length of the list corresponds to the number of recurrent blocks, the items in the
            list are the number of units of the LSTM layer in each block.

        dropout: float.
            Dropout rate to be applied after each recurrent block.
        '''
        
        super(LSTM, self).__init__()
        
        # check the inputs
        if type(units) != list:
            raise ValueError(f'The number of units should be provided as a list.')
        
        # build the model
        modules = OrderedDict()
        for i in range(len(units)):
            modules[f'LSTM_{i}'] = torch.nn.LSTM(
                input_size=input_length if i == 0 else units[i - 1],
                hidden_size=units[i],
                batch_first=True
            )
            modules[f'Lambda_{i}'] = Lambda(f=lambda x: x[0])
            modules[f'Dropout_{i}'] = torch.nn.Dropout(p=dropout)
        self.model = torch.nn.Sequential(modules)
    
    def forward(self, x):
        
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (samples, 1, length) where samples is the number of
            time series and length is the length of the time series.
        '''
        
        return self.model(x)[:, -1, :]


class FCN(torch.nn.Module):
    
    def __init__(self, filters, kernel_sizes):
    
        '''
        Parameters:
        __________________________________
        filters: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the number of filters (or channels) of the convolutional layer in each block.

        kernel_sizes: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the kernel sizes of the convolutional layer in each block.
        '''
        
        super(FCN, self).__init__()
        
        # check the inputs
        if len(filters) == len(kernel_sizes):
            blocks = len(filters)
        else:
            raise ValueError(f'The number of filters and kernel sizes must be the same.')

        # build the model
        modules = OrderedDict()
        for i in range(blocks):
            modules[f'Conv1d_{i}'] = torch.nn.Conv1d(
                in_channels=1 if i == 0 else filters[i - 1],
                out_channels=filters[i],
                kernel_size=(kernel_sizes[i],),
                padding='same'
            )
            modules[f'BatchNorm1d_{i}'] = torch.nn.BatchNorm1d(
                num_features=filters[i],
                eps=0.001,
                momentum=0.99
            )
            modules[f'ReLU_{i}'] = torch.nn.ReLU()
        self.model = torch.nn.Sequential(modules)
        
    def forward(self, x):
        
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (samples, 1, length) where samples is the number of
            time series and length is the length of the time series.
        '''
        
        return torch.mean(self.model(x), dim=-1)


class LSTM_FCN(torch.nn.Module):
    
    def __init__(self, input_length, units=[5,5], dropout=0.2, filters=[4,4], kernel_sizes=[3,3], num_classes=3):
        
        '''
        Parameters:
        __________________________________
        input_length: int.
            Length of the time series.

        units: list of int.
            The length of the list corresponds to the number of recurrent blocks, the items in the
            list are the number of units of the LSTM layer in each block.

        dropout: float.
            Dropout rate to be applied after each recurrent block.

        filters: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the number of filters (or channels) of the convolutional layer in each block.

        kernel_sizes: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the kernel sizes of the convolutional layer in each block.

        num_classes: int.
            Number of classes.
        '''
        
        super(LSTM_FCN, self).__init__()
        
        self.fcn = FCN(filters, kernel_sizes)
        self.lstm = LSTM(input_length, units, dropout)
        self.linear = torch.nn.Linear(in_features=filters[-1] + units[-1], out_features=num_classes)
    
    def forward(self, x):
        
        '''
        Parameters:
        __________________________________
        x: torch.Tensor.
            Time series, tensor with shape (samples, length) where samples is the number of time series
            and length is the length of the time series.
        
        Returns:
        __________________________________
        y: torch.Tensor.
            Logits, tensor with shape (samples, num_classes) where samples is the number of time series
            and num_classes is the number of classes.
        '''
        
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        y = torch.concat((self.fcn(x), self.lstm(x)), dim=-1)
        y = self.linear(y)
        
        return y


class Lambda(torch.nn.Module):
    
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f
    
    def forward(self, x):
        return self.f(x)


# import torch
# x= torch.rand([16, 768])
# model = LSTM_FCN(768)
# print(model(x).shape)

class FCN_model(nn.Module):
    def __init__(self,NumClassesOut,N_time,N_Features,N_LSTM_Out=128,N_LSTM_layers = 1
                 ,Conv1_NF = 128,Conv2_NF = 256,Conv3_NF = 128,lstmDropP = 0.8,FC_DropP = 0.3):
        super(FCN_model,self).__init__()
        
        self.N_time = N_time
        self.N_Features = N_Features
        self.NumClassesOut = NumClassesOut
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.lstm = nn.LSTM(self.N_Features,self.N_LSTM_Out,self.N_LSTM_layers, bidirectional=True)
        self.C1 = nn.Conv1d(self.N_Features,self.Conv1_NF,8)
        self.C2 = nn.Conv1d(self.Conv1_NF,self.Conv2_NF,5)
        self.C3 = nn.Conv1d(self.Conv2_NF,self.Conv3_NF,3)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        # self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out,self.NumClassesOut)
        self.FC = nn.Linear(384,self.NumClassesOut)
        self.device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    
    def init_hidden(self):
        
        h0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(self.device1)
        c0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(self.device1)
        return h0,c0
    
    def forward(self,x):
        
        # input x should be in size [B,T,F] , where B = Batch size
        #                                         T = Time sampels
        #                                         F = features
        
        # h0,c0 = self.init_hidden()
        # x1, (ht,ct) = self.lstm(x, (h0, c0))
        x1, (ht,ct) = self.lstm(x)
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x2))))
        x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2,2)
        
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.FC(x_all)
        return x_out