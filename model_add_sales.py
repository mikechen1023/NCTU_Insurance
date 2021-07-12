#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse
import matplotlib.pyplot as plt 
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
import time

# 顯示全部的 column
pd.set_option('display.max_columns', None)


# In[3]:


def DataAugumenting(data):
    """
    Goal:
        Data resample( duplicate "the successful insurance policy" )
    """
    # data augmentation
    y_data = data[data['是否已受理']==1]
    n_data = data[data['是否已受理']==0]

    print('處理前:')
    print('成交保單數 : ',len(y_data))
    print('未成交保單數 : ',len(n_data))
    print('保單總數 : ', len(data), end="\n\n")
    
    times = len(n_data)//len(y_data)

    new_y_data = y_data

    for i in range(times-1):
        new_y_data = pd.concat((new_y_data,y_data))

    new_data = pd.concat((new_y_data, n_data))

    print('處理後:')
    print('成交保單數 : ',len(new_y_data))
    print('未成交保單數 : ',len(n_data))
    print('保單總數 : ', len(new_data), end="\n\n")
    
    return new_data


# In[4]:


def DataPreprocessing(data, datatype, wealth_loyalty):
    """
    Input:
        param data : dataframe
        param datatype : new/old customer
        param wealth_loyalty : determine if put wealth and loyalty into datasets
    
    Output:
        x : data (dtype: torch)
        y : label (dtype: torch)
    """
    
    sales = pd.DataFrame(data, columns=['R1A', 'R1B', 'R1C', 'R2', 'R3', 'R4', 'R5', 'L1', 'L2', 'L3', 'L4', 'L5', 'other'])
    insurance = pd.DataFrame(data, columns=['保額', '保額區間', '保費', '保費區間', 'TWD'])
    commodity = pd.DataFrame(data, columns=['商品分類_三標'])
    age = pd.DataFrame(data, columns=['保險年齡'])
    y = pd.DataFrame(data, columns=['是否已受理'])
    
    # commodity use embedding
    type_dict = {'A&H':1, 'SP':2, 'RP1':3, 'RP2':4}
    embeds = nn.Embedding(5, 4) # 5 means dict dim : 0-4
    commodity = commodity.replace(type_dict)
    commodity = torch.from_numpy(commodity.to_numpy())
    commodity = torch.tensor(commodity)
    commodity_embedded = embeds(commodity)
    commodity = commodity_embedded.view(commodity_embedded.size(0),4).type(torch.DoubleTensor)
    print('commodity: ',commodity.dtype)
    
#     # commodity use one-hot encoding
#     commodity = pd.get_dummies(commodity)
#     commodity = torch.from_numpy(commodity.to_numpy())
    
    # transfer other data to torch type
    sales = torch.from_numpy(sales.to_numpy())
    insurance = torch.from_numpy(insurance.to_numpy())
    age = torch.from_numpy(age.to_numpy())
    y = torch.from_numpy(y.to_numpy())
    
    
    if datatype=='old':    #舊客戶
        if wealth_loyalty:  # 使用客戶忠誠度、財富指標
            level_dict = {'R1A':1, 'R1B':2, 'R1C':3, 'R2':4, 'R3':5, 'R4':6, 'R5':7}
            customer = pd.DataFrame(data, columns=['客戶忠誠度', '財富指標'])
            customer['客戶忠誠度'] = customer['客戶忠誠度'].str.get(1).astype(float)
            customer['財富指標'] = customer['財富指標'].replace(level_dict)
            customer = torch.from_numpy(customer.to_numpy())
            x = torch.cat((sales, insurance, age, customer, commodity), 1)
        else:
            customerLevel = pd.DataFrame(data, columns=['客戶分群(NEW)'])
            customerLevel = customerLevel['客戶分群(NEW)'].str.get(1).astype(float)
            customerLevel = torch.from_numpy(customerLevel.to_numpy())
            customerLevel = customerLevel.view(customerLevel.size(0), 1)
            print('sales: ',sales.dtype)
            print('insurance: ',insurance.dtype)
            print('customerLevel: ',customerLevel.dtype)
            x = torch.cat((sales, insurance, age, customerLevel, commodity), 1)
    else:    #新客戶
        x = torch.cat((sales, insurance, age, commodity), 1)
    
    return x,y


# In[5]:


def getData(csvpath, datatype, wealth_loyalty=False ,timesort=False):
    """
    Input:
        param csvpath : csv file path
        param datatype : new/old customer
        param wealth_loyalty : determine if put wealth and loyalty into datasets
        param timesort : determine if sort the datetime when split data into train/test datasets
        
    Output:
        x_train, y_train, x_test, y_test
    """
    
    # get dataframe
    df = pd.read_csv(csvpath, encoding="utf-8", index_col=[0])
    
    if datatype == 'new' or datatype=='all':
        data = pd.DataFrame(df, columns=['保額', '保額區間', '保費', '保費區間', '商品分類_三標', '保險年齡', 'TWD', '是否已受理', '建議書_建立日'])
    else:    # old              
        if wealth_loyalty:    # 使用忠誠度和財富指標
            data = pd.DataFrame(df, columns=['保額', '保額區間', '保費', '保費區間', '商品分類_三標', '保險年齡', 'TWD', '是否已受理', '財富指標', '客戶忠誠度', '建議書_建立日', 'R1A', 'R1B', 'R1C', 'R2', 'R3', 'R4', 'R5', 'L1', 'L2', 'L3', 'L4', 'L5', 'other']) 
        else:                 #使用客戶分群 
            data = pd.DataFrame(df, columns=['保額', '保額區間', '保費', '保費區間', '商品分類_三標', '保險年齡', 'TWD', '是否已受理', '客戶分群(NEW)', '建議書_建立日','R1A', 'R1B', 'R1C', 'R2', 'R3', 'R4', 'R5', 'L1', 'L2', 'L3', 'L4', 'L5', 'other']) #只用客戶分群
    
    # Delete the row that has NaN
    if datatype=='old':
        data = data.dropna()
    
    # Data Augument
    new_data = DataAugumenting(data)
    
    # sort datetime
    if timesort:
        # split data into train/test sets (split 8/2) with the sorting datetime
        new_data = new_data.sort_values(by = '建議書_建立日').reset_index(drop=True)
        train_data = new_data[0:int(len(new_data)*0.8)]
        test_data = new_data[int(len(new_data)*0.8):]
    else:
        # split data into train/test sets (random with 8/2)
        msk = np.random.rand(len(new_data)) < 0.8
        train_data = new_data[msk]
        test_data = new_data[~msk]
    
    # Data preprocessing
    x_train, y_train = DataPreprocessing(train_data, datatype, wealth_loyalty)
    x_test, y_test = DataPreprocessing(test_data, datatype, wealth_loyalty)
    
    return x_train, y_train, x_test, y_test


# In[6]:


def getdataloader(x, y, batch_size):
    """
    Goal : change torch to dataset
    
    """
    # change torch to dataset
    dataset = Data.TensorDataset(x, y)

    dataloader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        drop_last=True, 
        shuffle = True
    )
    
    return dataloader


# In[7]:


def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    acc = (rounded_preds == y).float()
    acc = acc.sum() / len(y)
    return acc


# In[8]:


# Model
class MyDNN(nn.Module):
    def __init__(self, model_args):
        super(MyDNN, self).__init__()
        self.args = model_args
        
        # init args
        self.input_dim = self.args.input_dim
        self.hidden_dim = self.args.hidden_dim
        self.output_dim = self.args.output_dim
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        output = self.layers(x.float())
        return output
    
    def printModel(self):
        print(model)


# In[9]:


class Mymodel(object):
    """
    Methods:
        - fit()        : Train the model 
        - predict()    : Output the prediction
        - evaluate()   : Calculate scores
        - save_model() : Save the model (.h5)
        
    Parameters:
        - epochs: int,
            Numer of iterations to run all data
            
        - batch_size: int,
            Minibatch size
            
        - learning rate: float,
            Initial learning rate
            
        - use_cuda: boolean,
            Run the model on gpu or cpu
            
        - optimizer: Algo of optimizer
        
        - criterion : Loss function
        
        - model_name: Model name
    
    """
    
    
    def __init__(self,
                 epochs=None, 
                 batch_size=None, 
                 learning_rate=None,
                 use_cuda=False, 
                 model_name="unknown_model.h5", 
                 model_args=None):
        
        # model related
        self._net = None
        self.model_args = model_args
        
        # learning related
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._net = MyDNN(self.model_args).to(self._device)
        self._optimizer = optim.SGD(self._net.parameters(), lr = self._learning_rate, momentum=0.9)
        self._criterion = nn.BCELoss()
        self._model_name = model_name
            
    def fit(self, trainloader):
        self._net.train()
        print(self._net)
        
        for epoch in range(self._epochs):
            
            epoch_loss = 0
            epoch_acc = 0
            for step, (batch_x, batch_y) in enumerate(trainLoader):
            
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                preds = self._net(batch_x.long())
                losses = self._criterion(preds, batch_y.float())
                acc = binary_accuracy(preds, batch_y)
                
                # Back prop.
                self._optimizer.zero_grad()
                losses.backward()
                self._optimizer.step()

                epoch_loss += losses.item()
                epoch_acc += acc
            print('epoch:{} | loss:{:4f} | acc:{:4f}'.format(epoch+1, epoch_loss/len(trainLoader), epoch_acc/len(trainLoader)))
            
    def predict(self, x_test):
        x_test = x_test.to(self._device)
        return self._net(x_test)
    
    def evaluate(self, x_test, y_test):
        x_test = x_test.to(self._device)
        
        self._net.eval()
        y_preds = self._net(x_test)
        y_preds = y_preds.cpu()
        y_preds = torch.round(y_preds)
        
        # 求 f1-score
        TP=0; FN=0; FP=0; TN=0
        for i in range(len(y_test)):
            if y_test[i]-y_preds[i]==1:
                FN += 1
            elif y_test[i]-y_preds[i]==-1:
                FP += 1
            else:
                if y_test[i]==1: TP += 1
                else: TN += 1

        print([[TP, FN], [FP, TN]])

        accuracy = (TP+TN)/len(y_test)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1_score = 2/((1/precision)+1/recall)

        print('len:',len(y_test))
        print('acc : ', accuracy)
        print('recall: ', recall)
        print('precision: ', precision)
        print('f1 score : ', f1_score)
    
    def save_model(self):
        torch.save(self._net, './model/' + self._model_name)


# In[10]:


if __name__ == '__main__':
    """
        input_dim:
            -新客戶: 10
            -舊客戶: 
                - 使用客戶分群: 24
                - 使用財富和忠誠度: 25
    """
    
    model_parser = argparse.ArgumentParser()
    
    # model dependent arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--input_dim', type=int, default=10)
    model_parser.add_argument('--hidden_dim', type=int, default=32)
    model_parser.add_argument('--output_dim', type=int, default=1)

    parser = argparse.ArgumentParser()
    
    # data/train arguments
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--use_cuda', type=bool, default=True)
    
    config = parser.parse_args(args=[])
    model_config = model_parser.parse_args(args=[])
    
        
    # data parameters
    csvpath = "./data/preprocessing_new.csv"
    datatype = 'new'
    timesort = False
    wealth_loyalty = False
    
    model_name = "test.h5"
    
    

    # load data
    x_train, y_train, x_test, y_test = getData(csvpath, datatype, wealth_loyalty, timesort)
    trainLoader = getdataloader(x_train, y_train, config.batch_size)
    
    print('x_train: ',x_train.shape)
    print('x_test: ',x_test.shape)
    
    model = Mymodel(epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    use_cuda=config.use_cuda, 
                    model_name=model_name, 
                    model_args=model_config)
    
    # fit model
    model.fit(trainLoader)
    
    # evaluate model
    model.evaluate(x_test, y_test)
    model.save_model()


# In[ ]:


# TP=12061; FN=12469; FP=12774; TN=18339
# accuracy = (TP+TN)/len(y_test)
# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# f1_score = 2/((1/precision)+1/recall)

# print('len:',len(y_test))
# print('acc : ', accuracy)
# print('recall: ', recall)
# print('precision: ', precision)
# print('f1 score : ', f1_score)


# In[ ]:




