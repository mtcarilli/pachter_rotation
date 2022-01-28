import numpy as np 
import matplotlib.pyplot as plt

import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy 
from scipy import optimize

class my_MLP1(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input = nn.Linear(input_dim, 256)
        self.hidden1 = nn.Linear(256, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, parameters):

        # pass parameters to the first layer of neurons 
        l_1 = self.input(parameters)

        # pass through a sigmoid function to second layer of neurons
        l_2 = torch.sigmoid(self.hidden1(l_1))
        
        # pass to a second layer of neurons 
        l_3 = torch.sigmoid((self.hidden2(l_2)))
        
        # pass to third layer of neurons 
        l_4 = (self.hidden3(l_3))

        # pass out to output dimensions (predicted weights), averaged to sum to 1 with softmax
        w_pred = self.softmax(self.output(l_4))

        return w_pred


def create_npy_file_paths(set_size,data_size = 102400,path_to_directory='../training_data_1D/'):
    '''Creates file paths for a certain set size. Stores in training_data'''
    file_paths = []
    num = int(data_size/set_size)
    
    for i in range(num):
        file_paths.append(path_to_directory+str(set_size)+'_'+str(i)+'_ker'+str(npdf)+'.npy')
        
    return(file_paths)

def load_data_list(file_path):
    full_file_list = list(np.load(file_path,allow_pickle=True))
    return(full_file_list)

def shuffle_training_data(full_file_list):
    '''Load .npy file, returns tensor for parameters, unweighted kernal functions, and ground truth histograms'''
    
    random.shuffle(full_file_list)
    parameters = np.array([ a[0] for a in full_file_list ])
    parameters_tensor = torch.from_numpy(parameters).float()
    yker_tensor = [ torch.tensor(a[1]).float() for a in full_file_list ]
    y_tensor = [ torch.tensor(a[2]).float() for a in full_file_list ]
    
    return(parameters_tensor,yker_tensor,y_tensor)

def get_probabilities_torch(yker, y, w):
    ''' Multiplies yker by weights, then reshapes Yker to be shape of y.'''
    # shapes of Yker and w?? 
    Y = torch.matmul(yker,w).reshape(y.shape)
    EPS=1e-8
    Y[Y<EPS]=EPS

    return Y

def get_metrics(yker,y,w,metric = 'kld'):
    '''Calculates desired metric between predicted Y and y.'''
    pred = get_probabilities_torch(yker,y,w)
    pred = pred.flatten()
    y = y.flatten()
    if metric=='totalse':
        return torch.sum((pred-y)**2)
    if metric=='mse':
        return torch.mean((pred-y)**2)
    if metric=='kld':
        # print(np.max(pred))
        # print(true*np.log(pred/true))
        return -torch.sum(y*torch.log(pred/y))
    if metric=='maxabsdev':
        return torch.max(torch.abs(pred-y))

def get_predicted_PMF(p_list,yker_list,y_list,position,model):
    '''Returns predicted histogram for p given current state of model.'''
    model.eval()

    p1 = p_list[position:position+1]
    w_p1 = model(p1)[0]
    predicted_y1 = get_probabilities_torch(yker_list[position],y_list[position],w_p1)
    
    return(predicted_y1)

    
    
def loss_fn(yker_list,y_list,w,batchsize,metric='kld'):
    '''Calculates average metval over batch between predicted Y and y.
    yker_list and y_list are actually lists of tensor histograms with first dimension batchsize'''
    
    metval = torch.tensor(0.0)
    
    for b in range(batchsize):
        y_ = y_list[b]
        yker_ = yker_list[b]
        w_ = w[b]
        met_ = get_metrics(yker_,y_,w_,metric=metric)
    
        
        metval += met_
    
    return(metval/batchsize)

# TESTING FUNCTIONS
def calculate_test_klds(test_list,model):
    parameters,yker_list,y_list = shuffle_training_data(test_list)
    metrics = np.zeros(len(parameters))
    
    for i in range(len(parameters)):
        Y = get_predicted_PMF(parameters,yker_list=yker_list,y_list=y_list,position=i,model=model)
        y = y_list[i]
        metric = -torch.sum(y*torch.log(Y/y))
        metrics[i] = metric.detach().numpy()
        
    metrics = np.array(metrics)
    return(metrics,np.mean(metrics))

def train(parameters_tensor,yker_list,y_list,model,optimizer,batchsize=64,metric = 'kld'):
    '''Trains the model for given input tensors and list of tensors. Divides training data into groups of 
    batchsizes. If the number of input parameters cannot be divided by batchsize, ignores remainder...'''
    
    metvals = []
    trials = int(np.floor(parameters_tensor.size()[0] / batchsize ))
    model.train()  # can this model be accessed inside the function ????? 
    
    for j in range(trials):
        i = j * batchsize
        p = parameters_tensor[i:i+batchsize]
        yker = yker_list[i:i+batchsize]
        y = y_list[i:i+batchsize]

        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        w_pred = model(p)

        # Compute loss
        loss = loss_fn(yker_list=yker,y_list=y,w=w_pred,batchsize=batchsize,metric=metric)
        
        metvals.append(loss.item())

        # Perform backward pass
        loss.backward()
      
        # Perform optimization
        optimizer.step()
    
    return(metvals)

def run_epoch_and_test(train_list,test_list,number_of_epochs,model,optimizer,batchsize=512,
                       metric = 'kld'):

    epoch_metrics = np.zeros(number_of_epochs)
    test_metrics = []
    batch_metrics_all = []

    for e in range(number_of_epochs):
        print('Epoch Number:',e)


        model.train()
        batch_metrics = []

        parameters,yker_list,y_list = shuffle_training_data(train_list)
        metric_ = train(parameters,yker_list,y_list,model=model,optimizer=optimizer,batchsize=batchsize,metric = metric)
        batch_metrics.append(metric_)
        batch_metrics_all.append(metric_)

        batch_metric_array = np.array(batch_metrics).flatten()
        epoch_metric_ = np.mean(batch_metric_array)

        epoch_metrics[e] = epoch_metric_


            # test by evaluating the model
        test_metric_list_,test_metric_ = calculate_test_klds(test_list,model=model)
        test_metrics.append(test_metric_)
    

    return(epoch_metrics,np.array(batch_metrics_all).flatten(),test_metrics)

def get_file_lists(npdf,number_of_training_files,number_of_testing_files):
    train_list = []
    test_list = []
    
    ker = npdf
    
    for i in range(number_of_training_files):
        num = i
        file_list_ = load_data_list(f'../training_data_1D/5120_{num}_ker{ker}.npy')
        train_list = train_list + file_list_

    for i in range(number_of_testing_files):
        num = 19-i
        file_list_ = load_data_list(f'../training_data_1D/5120_{num}_ker{ker}.npy')
        test_list = test_list + file_list_

    return(train_list,test_list)

def train_MLP1(train_list,test_list,num_epochs,batchsize,learning_rate=1e-3,
                     metric='kld'):
    
    model = my_MLP1(3,train_list[0][1].shape[1])      # define model 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # optimizer to use 

    e_,b_,t_ = run_epoch_and_test(train_list,test_list,
                                  num_epochs,optimizer=optimizer,batchsize=batchsize,model=model)
    
    return(e_,b_,t_,model)
