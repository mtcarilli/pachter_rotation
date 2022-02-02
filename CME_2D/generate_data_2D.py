import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import irfft2
import scipy
from scipy import integrate, stats
from numpy import linalg
import time
from scipy.special import gammaln
import numdifftools


def cme_integrator(p,lm,method='fixed_quad',fixed_quad_T=10,quad_order=60,quad_vec_T=np.inf):
    b,ki,bet,gam = p
    bet/=ki
    gam/=ki
    u = []
    mx = np.copy(lm)

    #initialize the generating function evaluation points
    mx[-1] = mx[-1]//2 + 1
    for i in range(len(mx)):
        l = np.arange(mx[i])
        u_ = np.exp(-2j*np.pi*l/lm[i])-1
        u.append(u_)
    g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
    for i in range(len(mx)):
        g[i] = g[i].flatten()[:,np.newaxis]

    #define function to integrate by quadrature.
    fun = lambda x: INTFUN(x,g,b,bet,gam)
    if method=='quad_vec':
        T = quad_vec_T*(1/bet+1/gam+1)
        gf = scipy.integrate.quad_vec(fun,0,T)[0]
    if method=='fixed_quad':
        T = fixed_quad_T*(1/bet+1/gam+1)
        gf = scipy.integrate.fixed_quad(fun,0,T,n=quad_order)[0]

    #convert back to the probability domain, renormalize to ensure non-negativity.
    gf = np.exp(gf) #gf can be multiplied by k in the argument, but this is not relevant for the 3-parameter input.
    gf = gf.reshape(tuple(mx))
    Pss = irfft2(gf, s=tuple(lm)) 
    EPS=1e-16
    Pss[Pss<EPS]=EPS
    Pss = np.abs(Pss)/np.sum(np.abs(Pss)) #always has to be positive...
    return Pss

def INTFUN(x,g,b,bet,gam):
    """
    Computes the Singh-Bokes integrand at time x. Used for numerical quadrature in cme_integrator.
    """
    if not np.isclose(bet,gam): #compute weights for the ODE solution.
        f = b*bet/(bet-gam)
        g[1] *= f
        g[0] *= b
        g[0] -= g[1]
        U = np.exp(-bet*x)*g[0]+np.exp(-gam*x)*g[1]
    else:
        g[1] *= (b*gam)
        g[0] *= b
        U = np.exp(-bet*x)*(g[0] + bet * g[1]* x)
    return U/(1-U)

def get_moments(p):
    b,ki,beta,gamma=p
    
    r = np.array([ki/beta, ki/gamma])
    MU = b*r
    VAR = MU*[1+b,1+b*beta/(beta+gamma)]
    STD = np.sqrt(VAR)
    xmax = np.ceil(MU+4*STD)
    xmax = np.clip(xmax,15,np.inf).astype('int')
    return MU, VAR, STD, xmax

def calculate_exact_cme(p,npdf,threshold=1e6,):
    '''Given parameter vector p, calculate the exact probabilites using CME integrator.'''
    p = np.insert(p,1,1) #set burst size to 1, somewhat arbitrary but input vector is 3-dim
    
    MU, VAR, STD, xmax = get_moments(p)
    
    # print(xmax) 
    # don't calculate if the output array is too big (greated than a million values)
    # can be 1000 per side
    if np.prod(xmax)>threshold:
        #print('Too big')
        return np.array([[-1],[-1]])
    
    y=cme_integrator(p,xmax+1)
    return(y)

def generate_grid(npdf,VAR,MU,method='logn'):

    lin = [np.linspace(0,1,npdf[i]+2)[1:-1] for i in range(2)]
    if method == 'logn': #if you want to use other basis function samples, generate ur own 
        logstd = np.sqrt(np.log((VAR/MU**2)+1))
        logmean = np.exp(np.log(MU)-logstd**2/2)
        translin = [stats.lognorm(scale=logmean[i], s=logstd[i]).ppf(lin[i]) for i in range(2)]
    if method == 'logn_tails':
        logstd = np.sqrt(np.log((VAR/MU**2)+1))
        logmean = np.exp(np.log(MU)-logstd**2/2)
        translin = [stats.lognorm(scale=logmean[i], s=logstd[i]).ppf(lin[i]) for i in range(2)]
        # print(np.asarray(translin).shape)
        # print(translin)
        translin = [np.asarray([MU[i]/4] + list(translin[i][1:-1]) + [2*MU[i]]) for i in range(2)]
        # print(translin)
    if method=='gamma': #trash
        a = MU**2/VAR
        scale = VAR/MU
        translin = [stats.gamma(scale=scale[i], a=a[i]).ppf(lin[i]) for i in range(2)]
    if method=='lin':
        translin = [lin[i]*(MU[i]+4*np.sqrt(VAR[i])) for i in range(2)]
    if method=='log':
        # translin = [np.logspace()
        translin = [(np.exp(lin[i])-1)/np.exp(1)*(MU[i]+4*np.sqrt(VAR[i])) for i in range(2)]
    if method=='exp': #trash
        # a = MU**2/VAR
        # scale = VAR/MU
        translin = [stats.expon(scale=MU[i]).ppf(lin[i]) for i in range(2)]
    # if method == 'weibull': #hard to implement
    #     coeff = [-0.220040320, -0.001433169, 0.150611381 , ]
    #     cv = np.sqrt(VAR)/MU
    #     wei_shape = 
    #     # translin[0][0] = MU[0]/4
    #     # translin[-1] = 2*MU
    return translin
    
def generate_kernel(p,xmax,npdf, hyp=1.2, calc_method = 'log', grid_method='logn', ax = None):
    MU, VAR, STD, xmax = get_moments(p)
    grid = generate_grid(npdf,VAR,MU,method=grid_method)
    if ax:
        g_ = np.meshgrid(*grid)
        g_ = [g_[i][:] for i in range(2)]
        ax.scatter(g_[0],g_[1],c='k',s=20)
    s = [hyp*np.insert(np.diff(grid[i]), 0, grid[i][0]) for i in range(2)]
    r = [(grid[i]/s[i])**2 for i in range(2)]
    p = [1/(1+s[i]**2/grid[i]) for i in range(2)]
    Y = np.zeros((np.prod(xmax+1),npdf[0],npdf[1]))
    xgrid = [np.arange(xmax[i]+1) for i in range(2)]

    if calc_method=='lin':
        for i in range(npdf[0]):
            Y_ = stats.nbinom.pmf(xgrid[0], r[0][i], p[0][i])
            for j in range(npdf[1]):
                Y[:,i,j] = np.outer(Y_,stats.nbinom.pmf(xgrid[1], r[1][j], p[1][j])).flatten()
    if calc_method=='log':
        for i in range(npdf[0]):
            Y_ = gammaln(xgrid[0]+r[0][i]) - gammaln(xgrid[0]+1) - gammaln(r[0][i]) + xgrid[0]*np.log(1-p[0][i]) + r[0][i]*np.log(p[0][i])
            for j in range(npdf[1]):
                Y[:,i,j] = (Y_[:,np.newaxis] + \
                                    gammaln(xgrid[1]+r[1][j]) - gammaln(xgrid[1]+1) - gammaln(r[1][j]) + xgrid[1]*np.log(1-p[1][j]) + r[1][j]*np.log(p[1][j])\
                                    ).flatten()
        Y = np.exp(Y) #this line goes from log pmf to real pmf... this might not actually be necessary... could we keep EVERYTHING in logspace? ...
    Y=Y.reshape((Y.shape[0],np.prod(npdf)))
    return Y


def wrap_kernels(p,npdf,hyp=2.4,grid_method='logn',metric='kld',w='lsq',threshold=1e6):

    p = np.insert(p,1,1) #this is somewhat arbitrary, but input is 3-dim
    MU, VAR, STD, xmax = get_moments(p)
    
    
    if np.prod(xmax)>threshold:
        return -1, -np.ones(np.prod(npdf))

    Yker = generate_kernel(p,xmax,npdf,hyp=hyp,grid_method=grid_method)
    
    return Yker


param_vectors = np.load('../training_data_2D/gennady_parameters.npy',allow_pickle=True)
param_vectors = 10**param_vectors

def create_file_paths(set_size,npdf,data_size = 102400,path_to_directory='../training_data_2D/'):
    '''Creates file paths for a certain set size. Stores in t'''
    file_paths = []
    num = int(data_size/set_size)
    
    for i in range(num):
        file_paths.append(path_to_directory+str(set_size)+'_'+str(i)+'_ker'+str(npdf))
        
    return(file_paths)


def prepare_set(position,size,npdf,param_vectors=param_vectors):
    '''Outputs exact CME, unweighted kernels, and parameters for a batchsized amount of parameters.'''
    set_list = []
    
    for i in range(size):
        param_ = param_vectors[i+position]
        y_ = calculate_exact_cme(param_,npdf=[npdf,npdf])
        if y_[0][0] != -1:
            set_i_ = []
            yker_ = wrap_kernels(p=param_,npdf=[npdf,npdf])
            set_i_.append(np.array(param_))
            set_i_.append(np.array(yker_))
            set_i_.append(np.array(y_))
            set_list.append(set_i_)
        else: 
            continue

    return(set_list)

def save_set(file_path,position,set_size,npdf,param_vectors=param_vectors):
    '''Prepares and saves a set of params, kernels, groud truth
    at position of given size. So define that, dear.'''
    set_list = prepare_set(position=position,size = set_size,npdf=npdf,param_vectors=param_vectors)

    np.save(file_path,set_list)
    
    
def generate_sets(set_size,npdf,data_size = 102400,path_to_directory='../training_data_2D/',param_vectors=param_vectors):
    '''Generates kernel and true histograms for params in param_vectors
    Saves them in sets of set_size, in directory path_to_directory.'''
    
    file_paths = create_file_paths(set_size,npdf=npdf,data_size = data_size,path_to_directory=path_to_directory)
    
    for i,file in enumerate(file_paths):
        print(i)
        position = i*set_size
        save_set(file,position,set_size,npdf=npdf,param_vectors=param_vectors)
