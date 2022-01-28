import numpy as np
import random 
import matplotlib.pyplot as plt
import torch
import train_MLP1 as tm1

def get_predicted_PMF(p_list,yker_list,y_list,position,model):
    '''Returns predicted histogram for p given current state of model.'''
    model.eval()
    
    p1 = p_list[position:position+1]

    w_p1 = model(p1)[0]
    predicted_y1 = get_probabilities_torch(yker_list[position],y_list[position],w_p1)
    
    return(predicted_y1)

def get_probabilities_torch(yker, y, w):
    ''' Multiplies yker by weights, then reshapes Yker to be shape of y.'''
    # shapes of Yker and w?? 
    Y = torch.matmul(yker,w).reshape(y.shape)
    EPS=1e-8
    Y[Y<EPS]=EPS

    return Y

def plot_PMF_grid(file_list,nrows,ncols,model,kld=True):
    '''Plots predicted and true PMFs for random parameters chosen from file_list.
    Number: nrows*ncols'''
    p_list,yker_list,y_list = tm1.shuffle_training_data(file_list)
    
    npdf = yker_list0[0].shape[1]
    
    rand = np.zeros(nrows*ncols)
    
    for i in range(nrows*ncols):
        rand[i] = random.randint(0,len(y_list))
    
    y = []
    Y = []
    
    for r in rand:
        r = int(r)
        y_pred = get_predicted_PMF(p_list=p_list,
                                yker_list=yker_list,y_list=y_list,position=r,model=model)
        
        y.append(y_list[r])
        Y.append(y_pred)
    
    Y = [Y_.detach().numpy() for Y_ in Y]
    y = [y_.detach().numpy() for y_ in y]
    


    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))


    for i in range(nrows):
        for j in range(ncols):
            k = i+j
            x = np.arange(len(y[k]))
            axs[i,j].plot(x,y[k],'k-',label='True PMF')
            axs[i,j].plot(x,Y[k],'r--',label='Predicted PMF')
        
            axs[i,j].set_xlabel('# mat RNA')
            axs[i,j].set_ylabel('probability')

            axs[i,j].legend()
            
            if kld == True:
                kld_ = -np.sum(y[k]*np.log(Y[k]/y[k]))
                axs[i,j].title.set_text(f'KLD: {kld_}')
        
    fig.tight_layout()


def calculate_test_klds(test_list,model):
    '''Calculates klds for parameters given current state of model'''
    parameters,yker_list,y_list = tm1.shuffle_training_data(test_list)
    metrics = np.zeros(len(parameters))
    
    for i in range(len(parameters)):
        Y = get_predicted_PMF(parameters,yker_list=yker_list,y_list=y_list,
                              position=i,model=model)
        y = y_list[i]
        metric = -torch.sum(y*torch.log(Y/y))
        metrics[i] = metric.detach().numpy()
        
    metrics = np.array(metrics)
    return(metrics,np.mean(metrics))


def plot_CDF(array,npdf=None,epochs=None,xlim=None):
    '''Plots CDF'''
    cdf = np.zeros(len(array))
    array_sorted = np.sort(array)
    for i,value in enumerate(array_sorted):
        cdf[i] = len(array_sorted[array_sorted<value])/len(array_sorted)
        
    plt.figsize(10,10)
    plt.scatter(array,cdf,s=5)
    plt.xlabel('KL Divergence')
    plt.ylabel('CDF')
    
    if npdf:
        plt.title(f'CDF of KLDs for {epochs} epochs, NPDF = {npdf}')
    if xlim:
        xlow,xhigh=xlim
        plt.xlim(xlow,xhigh)
        
    plt.show()


def plot_histogram(array,bins,xlim=None):
    '''Histogram of bin number of bins, xlim'''
    plt.hist(array,bins = bins)
    if xlim:
        xlow,xhigh = xlim
        plt.xlim(xlow,xhigh)
    plt.title(f'Max KLD: {np.max(klds_currently)}, Min KLD: {np.min(klds_currently)}')
    plt.xlabel('KL Divergence')
    plt.ylabel('Frequency')
    
def plot_training(e_,t_,npdf=None,batchsize=512):
    '''Plots training data'''
    plt.figure(figsize=(9,6))
    plt.plot(range(len(e_)),e_,c='blue',label='Training Data')
    plt.plot(range(len(t_)),t_,c='red',label='Testing Data')
    plt.suptitle(f'Min KLD: {np.min(e_)}')
    if npdf:
        plt.title(f'NPDF = {npdf}, Batchsize = {batchsize}')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.legend()

def get_parameters_high_kld(train_list,model,percent=5):
    '''Returns given percent parameters with the highest klds and klds.'''
    klds,kld_mean = calculate_test_klds(test_list,model)
    parameters,yker_list,y_list = tm1.shuffle_training_data(test_list)
    
    num_params = int(np.floor(len(test_list)*percent*10**-2))
    
    ind = np.argpartition(klds, -num_params)[-num_params:]
    
    high_klds = klds[ind]
    high_params = parameters[ind]
    
    return(high_params,high_klds)


def plot_high_params(high_params,high_klds):
    '''Plots the params b vs. gamma, b vs. beta and beta vs. gamma'''
    high_b = 10**np.array([ p[0] for p in high_params ])
    high_beta = 10**np.array([ p[1] for p in high_params ])
    high_gamma = 10**np.array([ p[2] for p in high_params ])
    
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(10,5))
    
    ax[0].scatter(high_b,high_beta,c = high_klds,cmap='Purples')
    ax[0].set_xlabel('b')
    ax[0].set_ylabel('beta')
    
    ax[1].scatter(high_b,high_gamma,c = high_klds,cmap='Purples')
    ax[1].set_xlabel('b')
    ax[1].set_ylabel('gamma')
    
    ax[2].scatter(high_beta,high_gamma,c = high_klds,cmap='Purples')
    ax[2].set_xlabel('beta')
    ax[2].set_ylabel('gama')
    
    fig.tight_layout()

