#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
pytorch == 1.5.0
cuda == 10.1
python == 3.8
pandas == 0.25.0
numpy == 1.18.4
argparse == 1.1
matplotlib == 3.1.1
"""


# In[1]:


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


# In[2]:


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


# In[3]:


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
    
    insurance = pd.DataFrame(data, columns=['保額', '保額區間', '保費', '保費區間', 'TWD'])
    commodity = pd.DataFrame(data, columns=['商品分類_三標'])
    age = pd.DataFrame(data, columns=['保險年齡'])
    y = pd.DataFrame(data, columns=['是否已受理'])
    
    # commodity use embedding
    type_dict = {'A&H':1, 'SP':2, 'RP1':3, 'RP2':4}
    embeds = nn.Embedding(5, 4) # 5 means dict dim : 0-4
    commodity = commodity.replace(type_dict)
    commodity = torch.from_numpy(commodity.to_numpy())
    commodity = torch.tensor(commodity, dtype=torch.long)
    commodity_embedded = embeds(commodity)
    commodity = commodity_embedded.view(commodity_embedded.size(0),4)


#     # commodity use one-hot encoding
#     commodity = pd.get_dummies(commodity)
#     commodity = torch.from_numpy(commodity.to_numpy())
    
    # transfer other data to torch type
    insurance = torch.from_numpy(insurance.to_numpy())
    age = torch.from_numpy(age.to_numpy())
    y = torch.from_numpy(y.to_numpy())
    
    
    if datatype=='old':    #舊客戶
        if wealth_loyalty:  # 使用客戶忠誠度、財富指標
            level_dict = {'R1A':1, 'R1B':2, 'R1C':3, 'R2':4, 'R3':5, 'R4':6, 'R5':7}
            customer = pd.DataFrame(data, columns=['客戶忠誠度', '財富指標'])
            customer['客戶忠誠度'] = customer['客戶忠誠度'].str.get(1).astype(int)
            customer['財富指標'] = customer['財富指標'].replace(level_dict)
            customer = torch.from_numpy(customer.to_numpy())
            x = torch.cat((insurance, age, customer, commodity), 1)
        else:
            customerLevel = pd.DataFrame(data, columns=['客戶分群(NEW)'])
            customerLevel = customerLevel['客戶分群(NEW)'].str.get(1).astype(int)
            customerLevel = torch.from_numpy(customerLevel.to_numpy())
            customerLevel = customerLevel.view(customerLevel.size(0), 1)
            x = torch.cat((insurance, age, customerLevel, commodity), 1)
    else:    #新客戶
        x = torch.cat((insurance, age, commodity), 1)
    
    return x,y


# In[4]:


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
            data = pd.DataFrame(df, columns=['保額', '保額區間', '保費', '保費區間', '商品分類_三標', '保險年齡', 'TWD', '是否已受理', '財富指標', '客戶忠誠度', '建議書_建立日']) 
        else:                 #使用客戶分群 
            data = pd.DataFrame(df, columns=['保額', '保額區間', '保費', '保費區間', '商品分類_三標', '保險年齡', 'TWD', '是否已受理', '客戶分群(NEW)', '建議書_建立日']) #只用客戶分群

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


# In[5]:


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


# In[6]:


def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    acc = (rounded_preds == y).float()
    acc = acc.sum() / len(y)
    return acc


# In[7]:


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


# In[8]:


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


# In[9]:


if __name__ == '__main__':
    """
        input_dim:
            -新客戶: 10
            -舊客戶: 
                - 使用客戶分群: 11
                - 使用財富和忠誠度: 12
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


# In[18]:


TP=12061; FN=12469; FP=12774; TN=18339
accuracy = (TP+TN)/len(y_test)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1_score = 2/((1/precision)+1/recall)

print('len:',len(y_test))
print('acc : ', accuracy)
print('recall: ', recall)
print('precision: ', precision)
print('f1 score : ', f1_score)


# In[63]:


# model.fit(trainLoader)


# In[ ]:


# model.evaluate(x_test, y_test)


# In[14]:


# def fit(data, model, optimizer, criterion, batch_size, epochs, use_cuda=False):
    
#     # run model on a GPU or CPU
#     device = torch.device("cuda" if use_cuda else "cpu")
#     model._device = device
    
#     # set model to training mode
#     model.train()
    
#     for epoch in range(epochs):
        
#         epoch_loss = 0
#         epoch_acc = 0
        
#         for step, (batch_x, batch_y) in enumerate(trainLoader):
            
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)

#             preds = model(batch_x.long())
#             losses = criterion(preds, batch_y.float())
#             acc = binary_accuracy(preds, batch_y)

#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()

#             epoch_loss += losses.item()
#             epoch_acc += acc
#         print('epoch:{} | loss:{:4f} | acc:{:4f}'.format(epoch+1, epoch_loss/len(trainLoader), epoch_acc/len(trainLoader)))


# In[15]:


# def evaluate(model, x_test, y_test):
    
#     # set model to evaulating mode
#     model.eval()
    
#     x_test = x_test.to(model._device)
    
#     # predict
#     y_preds = model(x_test)
#     y_preds = y_preds.cpu()
#     y_preds = torch.round(y_preds)
    
#     # 求 f1-score
#     TP=0; FN=0; FP=0; TN=0
#     for i in range(len(y_test)):
#         if y_test[i]-y_preds[i]==1:
#             FN += 1
#         elif y_test[i]-y_preds[i]==-1:
#             FP += 1
#         else:
#             if y_test[i]==1: TP += 1
#             else: TN += 1

#     print([[TP, FN], [FP, TN]])

#     accuracy = (TP+TN)/len(y_test)
#     precision = TP/(TP+FP)
#     recall = TP/(TP+FN)
#     f1_score = 2/((1/precision)+1/recall)

#     print('len:',len(y_test))
#     print('acc : ',accuracy)
#     print('recall: ',recall)
#     print('f1 score : ',f1_score)


# In[ ]:


# # 建立 model
# model = DNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# # 設定 backprop.參數和 loss function
# optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
# criterion = nn.BCELoss()

# # 設定 GPU 運算
# device = torch.device('cuda')

# model = model.to(device)
# criterion = criterion.to(device)


# In[ ]:


print(model)


# In[ ]:


y_train.shape


# In[ ]:


# # train
# model.train()
# for epoch in range(EPOCHS):
    
#     epoch_loss = 0
#     epoch_acc = 0
    
#     for step, (batch_x, batch_y) in enumerate(trainLoader):
        
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.to(device)
        
#         preds = model.forward(batch_x.long())
#         losses = criterion(preds, batch_y.float())
#         acc = binary_accuracy(preds, batch_y)
        
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
        
#         epoch_loss += losses.item()
#         epoch_acc += acc
#     print('epoch:{} | loss:{:4f} | acc:{:4f}'.format(epoch+1, epoch_loss/len(trainLoader), epoch_acc/len(trainLoader)))


# In[ ]:


# # evaulate
# model.eval()
# x_test = x_test.to(device)

# y_preds = model(x_test)

# y_preds = y_preds.cpu()
# y_preds = torch.round(y_preds)

# # 求 f1-score
# TP=0; FN=0; FP=0; TN=0
# for i in range(len(y_test)):
#     if y_test[i]-y_preds[i]==1:
#         FN += 1
#     elif y_test[i]-y_preds[i]==-1:
#         FP += 1
#     else:
#         if y_test[i]==1: TP += 1
#         else: TN += 1

# print([[TP, FN], [FP, TN]])

# accuracy = (TP+TN)/len(y_test)
# precision = TP/(TP+FP)
# recall = TP/(TP+FN)
# f1_score = 2/((1/precision)+1/recall)

# print('len:',len(y_test))
# print('acc : ',accuracy)
# print('f1 score : ',f1_score)

# cnt=0
# for i in y_preds:
#     if i.item()==1:
#         cnt+=1

# print('cnt: ',cnt)
# print('total: ',len(y_preds))


# In[ ]:


# # load data

# CSVPATH = "./preprocessing_v3_new.csv"
# DATATYPE = 'new'
# TIMESORT = True


# x_train, y_train, x_test, y_test = getData(CSVPATH, DATATYPE, TIMESORT)

# print('x_train: ', x_train.shape)
# print('y_train: ', y_train.shape)
# print('x_test: ', x_test.shape)
# print('y_test: ', y_test.shape)


# In[ ]:


# INPUT_DIM = 8
# HIDDEN_DIM = 32
# OUTPUT_DIM = 1
# BATCH_SIZE = 64
# EPOCHS = 10


# In[ ]:


# # change torch to dataset
# train_dataset = Data.TensorDataset(x_train, y_train)
# test_dataset = Data.TensorDataset(x_test, y_test)

# trainLoader = Data.DataLoader(
#     dataset = train_dataset,
#     batch_size = BATCH_SIZE,
#     shuffle = True
# )

# testLoader = Data.DataLoader(
#     dataset = test_dataset,
#     shuffle = False
# )


# In[ ]:


# """
#     只用客戶分群
# """
# def DataPreprocessing(data, datatype):
    
#     insurance = pd.DataFrame(data, columns=['保額', '保費', 'TWD'])
#     commodity = pd.DataFrame(data, columns=['商品分類_三標'])
#     age = pd.DataFrame(data, columns=['保險年齡'])
#     y = pd.DataFrame(data, columns=['是否已受理'])
    
#     # set embedding dictionary
#     type_dict = {'A&H(健康意外險)':0, 'SP(躉繳)':1, '躉繳':1, 'RP金流(壽險期繳金流型)':2, '終身壽險':2, 'RP保障(壽險期繳保障型)':3}
#     embeds = nn.Embedding(4, 4)
    
#     commodity = commodity.replace(type_dict)
#     commodity = torch.from_numpy(commodity.to_numpy())
#     commodity = torch.tensor(commodity, dtype=torch.long)
#     commodity_embedded = embeds(commodity)
    
#     # transfer other data to torch type
#     insurance = torch.from_numpy(insurance.to_numpy())
#     age = torch.from_numpy(age.to_numpy())
#     y = torch.from_numpy(y.to_numpy())
    
#     commodity_embedded = commodity_embedded.view(commodity_embedded.size(0),4)
    
#     if datatype=='old':
#         customerLevel = pd.DataFrame(data, columns=['客戶分群(NEW)'])
#         customerLevel = customerLevel['客戶分群(NEW)'].str.get(1).astype(int)
#         customerLevel = torch.from_numpy(customerLevel.to_numpy())
#         customerLevel = customerLevel.view(customerLevel.size(0), 1)
#         x = torch.cat((insurance, age, customerLevel, commodity_embedded), 1)
#     else:
#         x = torch.cat((insurance, age, commodity_embedded), 1)
    
#     return x,y


# In[ ]:





# In[ ]:


# # test dataloader
# for epoch in range(5):
#     i=0
#     for batch_x, batch_y in loader:
#         i = i + 1
#         print('Epoch:{} | num:{} | batch_x:{} | batch_y:{}'.format(epoch, i, batch_x, batch_y))

# dataloader 使用
# for epoch in range(1):
#     i = 0
#     for step, (batch_x, batch_y) in enumerate(loader):
#         i += 1
#         print('Epoch:{} | num:{} | batch_x:{} | batch_y:{}'.format(epoch, i, batch_x, batch_y), end="\n\n")


# In[ ]:


# # split data to train/test sets
# msk = np.random.rand(len(new_data)) < 0.8
# train_data = new_data[msk]
# test_data = new_data[~msk]

# print(len(new_data))
# print(len(train_data))
# print(len(test_data))

# # get train data
# insurance_train = pd.DataFrame(train_data, columns=['保額', '保費', 'TWD'])
# commodity_train = pd.DataFrame(train_data, columns=['商品分類_三標'])
# person_train = pd.DataFrame(train_data, columns=['保險年齡'])
# y_train = pd.DataFrame(train_data, columns=['是否已受理'])


# # get test data
# insurance_test = pd.DataFrame(test_data, columns=['保額', '保費', 'TWD'])
# commodity_test = pd.DataFrame(test_data, columns=['商品分類_三標'])
# person_test = pd.DataFrame(test_data, columns=['保險年齡'])
# y_test = pd.DataFrame(test_data, columns=['是否已受理'])


# '''
# embed the commodity data
# '''

# # set embedding dictionary
# type_dict = {'A&H(健康意外險)':0, 'SP(躉繳)':1, '躉繳':1, 'RP金流(壽險期繳金流型)':2, '終身壽險':2, 'RP保障(壽險期繳保障型)':3}
# embeds = nn.Embedding(4, 4)

# # embedding
# commodity_train = commodity_train.replace(type_dict)
# commodity_train = torch.from_numpy(commodity_train.to_numpy())
# commodity_test = commodity_test.replace(type_dict)
# commodity_test = torch.from_numpy(commodity_test.to_numpy())

# # transfer type from numpy to torch tensor (dtype = torch.long)
# commodity_train = torch.tensor(commodity_train, dtype=torch.long)
# commodity_train_embedded = embeds(commodity_train)
# commodity_test = torch.tensor(commodity_test, dtype=torch.long)
# commodity_test_embedded = embeds(commodity_test)

# # transfer other data to torch type
# insurance_train = torch.from_numpy(insurance_train.to_numpy())
# person_train = torch.from_numpy(person_train.to_numpy())
# y_train = torch.from_numpy(y_train.to_numpy())

# insurance_test = torch.from_numpy(insurance_test.to_numpy())
# person_test = torch.from_numpy(person_test.to_numpy())
# y_test = torch.from_numpy(y_test.to_numpy())

# # check data type
# print('insurance_train: ', insurance_train.shape)
# print('person_train', person_train.shape)
# print('commodity_train_embedded', commodity_train_embedded.shape)

# # concate the data
# commodity_train_embedded = commodity_train_embedded.view(commodity_train_embedded.size(0),4)
# x_train = torch.cat((insurance_train, person_train, commodity_train_embedded), 1)

# commodity_test_embedded = commodity_test_embedded.view(commodity_test_embedded.size(0), 4)
# x_test = torch.cat((insurance_test, person_test, commodity_test_embedded), 1)

# # check data type
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)


# In[ ]:





# In[ ]:


# # 測試 成交保單 的準確率

# data_access = pd.DataFrame(df, columns=['保額', '保費', '商品分類_三標', '保險年齡', 'TWD', '是否已受理'])
# data_access = data_access.dropna()

# data_access['保額'] = data_access['保額'] * 100
# data_access['保費'] = data_access['保費'] * 100

# data_access = data_access[data_access['是否已受理'] == 0]

# insurance_access = pd.DataFrame(data_access, columns=['保額', '保費', 'TWD'])
# commodity_access = pd.DataFrame(data_access, columns=['商品分類_三標'])
# person_access = pd.DataFrame(data_access, columns=['保險年齡'])
# y_access = pd.DataFrame(data_access, columns=['是否已受理'])

# type_dict = {'A&H(健康意外險)':0, 'SP(躉繳)':1, '躉繳':1, 'RP金流(壽險期繳金流型)':2, '終身壽險':2, 'RP保障(壽險期繳保障型)':3}
# embeds = nn.Embedding(4, 4)

# commodity_access = commodity_access.replace(type_dict)
# commodity_access = torch.from_numpy(commodity_access.to_numpy())
# commodity_access = torch.tensor(commodity_access, dtype=torch.long)
# commodity_access_embedded = embeds(commodity_access)

# insurance_access = torch.from_numpy(insurance_access.to_numpy())
# person_access = torch.from_numpy(person_教師資格考試access.to_numpy())
# y_access = torch.from_numpy(y_access.to_numpy())

# commodity_access_embedded = commodity_access_embedded.view(commodity_access_embedded.size(0),4)
# x_access = torch.cat((insurance_access, person_access, commodity_access_embedded), 1)

# x_access = x_access.to(device)
# y_access = y_access.to(device)


# In[ ]:


preds = model(x_access)


# In[ ]:


binary_accuracy(preds, y_access)


# In[ ]:


count = 0
for i in preds:
    if i>0.5:
        count+=1
        
print(count)


# In[ ]:


print(count/len(preds))

