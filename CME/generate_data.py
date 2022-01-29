import numpy as np 
from scipy.fft import irfft
import scipy
from scipy import integrate, stats
from numpy import linalg
import time
from scipy.special import gammaln
from numpy.random import normal


def cme_integrator(p,lm,method='fixed_quad',fixed_quad_T=10,quad_order=60,quad_vec_T=np.inf):
    b,bet,gam = p

    #initialize the generating function evaluation points
    mx = lm//2 + 1
    l = np.arange(mx)
    g = np.exp(-2j*np.pi*l/lm)-1

    #define function to integrate by quadrature.
    # print(g[1])
    g = g[:,np.newaxis]
    # print(g[1])
    # print(g.shape)
    fun = lambda x: INTFUN(x,g,b,bet,gam)
    if method=='quad_vec':
        T = quad_vec_T*(1/bet+1/gam+1)
        gf = scipy.integrate.quad_vec(fun,0,T)[0]
    if method=='fixed_quad':
        # print('begin fix quad')
        T = fixed_quad_T*(1/bet+1/gam+1)
        # print(g[1])
        gf = scipy.integrate.fixed_quad(fun,0,T,n=quad_order)[0]
        # print(g.shape)
        # print(gf.shape)
        # print(g[1])
        # print(gf[1])
        # print(T)
        # print('end fix quad')
    if method=='iteration':
        # print('begin iter')

        T_ = fixed_quad_T*(1/bet+1/gam+1)
        # print(T_)
        # print(g[1])
        # print(g.shape)
        g=np.squeeze(g)
        # print(g[1])
        
        fun = lambda x: INTFUN(x,g,b,bet,gam)
        gf = np.zeros(g.shape,dtype='complex128')
        n=quad_order
        # print(g[1])
        d = T_/n
        # print(d*1-0.5*d)
        # print(fun(d*30-0.5*d))
        for i in range(1,n+1):
            e = fun(d*i-0.5*d)*d
            gf += e
            # print(e[1])
        # print(gf[1])
        # print('end iter')
        # print(g.shape)
        # print(gf.shape)
    gf = np.exp(gf) #gf can be multiplied by k in the argument, but this is not relevant for the 3-parameter input.
    Pss = irfft(gf, n=lm) 
    EPS=1e-16
    Pss[Pss<EPS]=EPS
    Pss = np.abs(Pss)/np.sum(np.abs(Pss)) #always has to be positive...
    return Pss

def INTFUN(x,g_,b,bet,gam):
    """
    Computes the Singh-Bokes integrand at time x. Used for numerical quadrature in cme_integrator.
    """
    if not np.isclose(bet,gam): #compute weights for the ODE solution.
        f = b*bet/(bet-gam)
        # g_ *= f
        U = g_*f*(-np.exp(-bet*x)+np.exp(-gam*x))
    else:
        # g_ *= ()
        U = np.exp(-bet*x)*(bet * g_ *b* x)
    return U/(1-U)

def get_moments(p):
    b,beta,gamma=p
    
    r = np.array([1/beta, 1/gamma])
    MU = b*r
    VAR = MU*[1+b,1+b*beta/(beta+gamma)]
    STD = np.sqrt(VAR)
    xmax = np.ceil(MU+4*STD)
    xmax = np.maximum(15,xmax).astype('int')
    xmax = xmax[1]
    MU=MU[1]
    VAR=VAR[1]
    STD=STD[1]
    return MU, VAR, STD, xmax


def calculate_exact_cme(p, threshold=1e6,npdf = [5,5]):
    '''Given parameter vector p, calculate the exact probabilites using CME integrator.'''
    
    MU, VAR, STD, xmax = get_moments(p)
    
    
    y=cme_integrator(p,xmax+1)
    return(y)

def generate_grid(npdf,VAR,MU,method='logn',quantiles='lin'):
    if quantiles=='lin':
        q = np.linspace(0,1,npdf+2)[1:-1]
    elif quantiles=='cheb':
        n = np.arange(npdf)
        q = (np.cos((2*(n+1)-1)/(2*npdf)*np.pi)+1)/2
    elif quantiles=='PRESET':
        logstd = np.sqrt(np.log((VAR/MU**2)+1))
        logmean = np.log(MU**2/np.sqrt(VAR+MU**2))
        translin = np.exp(logmean+logstd*NORM)
        return translin
    if method == 'logn': #if you want to use other basis function samples, generate ur own 
        norm = stats.norm.ppf(q)
        logstd = np.sqrt(np.log((VAR/MU**2)+1))
        logmean = np.log(MU**2/np.sqrt(VAR+MU**2))
        translin = np.exp(logmean+logstd*norm)
    if method=='lin':
        translin=q*(MU+3*np.sqrt(VAR))
    # print(translin)
    return translin
    
def generate_kernel(p,xmax,npdf, hyp=1.2, calc_method = 'log', grid_method='logn', ax = None,kernel='nb',quantiles='lin'):
    MU, VAR, STD, xmax = get_moments(p)
    grid = generate_grid(npdf,VAR,MU,method=grid_method,quantiles=quantiles)
    if ax:
        ax.scatter(grid,np.zeros(npdf),c='k',s=20)
    s = hyp*np.insert(np.diff(grid), 0, grid[0])
    r = (grid/s)**2 
    p = 1/(1+s**2/grid)
    Y = np.zeros((xmax+1,npdf))
    xgrid = np.arange(xmax+1) 

    if kernel == 'nb':
        if calc_method=='log':
            for i in range(npdf):
                Y[:,i] = gammaln(xgrid+r[i]) - gammaln(xgrid+1) - gammaln(r[i]) + xgrid*np.log(1-p[i]) + r[i]*np.log(p[i])
            Y = np.exp(Y) 
    return Y


def wrap_kernels_1D(p,npdf,hyp=2.4,grid_method='logn',metric='kld',w='lsq',threshold=1e6):
    'Returns kernels or, in this case, vector for the 1D PMF.'
    # p = np.insert(p,1,1) #this is somewhat arbitrary, but input is 3-dim
    MU, VAR, STD, xmax = get_moments(p)
    
    
    if np.prod(xmax)>threshold:
        return -1, -np.ones(np.prod(npdf))

    Yker = generate_kernel(p,xmax,npdf,hyp=hyp,grid_method=grid_method)
    
    return Yker


def generate_param_vectors(N):
    '''Generates N parameter vectors randomly spaced in logspace between bounds.'''
    
    logbnd = np.log10([[1,300],[0.05,50],[0.05,50]])
    dbnd = logbnd[:,1]-logbnd[:,0]
    lbnd = logbnd[:,0]
    param_vectors = np.zeros((N,3))


    i = 0
    while i<N:
        th = np.random.rand(3)*dbnd+lbnd
        MU, VAR, STD, xmax=get_moments(10**th)

        if xmax>1e6:
            continue
        if th[1] == th[2]:
            continue
        else:
            param_vectors[i,:] = th
            i+=1
            
    return(param_vectors)

param_vectors = np.load('../training_data_1D/gennady_parameters.npy',allow_pickle=True)
param_vectors = 10**param_vectors

def create_file_paths(set_size,npdf,data_size = 102400,path_to_directory='./training_data_1D/'):
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
        y_ = calculate_exact_cme(param_)
        set_i_ = []
        yker_ = wrap_kernels_1D(p=param_,npdf=npdf)
        set_i_.append(np.log10(np.array(param_)))
        set_i_.append(np.array(yker_))
        set_i_.append(np.array(y_))
        set_list.append(set_i_)

    return(set_list)

def save_set(file_path,position,set_size,npdf,param_vectors=param_vectors):
    '''Prepares and saves a set at position of given size. All depends on overall
    parameter vector list. Now stores parameters as the log10 of raw parameters.. dunno why tbh'''
    set_list = prepare_set(position=position,npdf=npdf,size = set_size,param_vectors=param_vectors)

    np.save(file_path,set_list)
    
    
def generate_sets(set_size,npdf,data_size = 102400,path_to_directory='./training_data_1D/',param_vectors=param_vectors):
    '''Generates kernel and true histograms for params in param_vectors
    Saves them in sets of set_size, in directory path_to_directory.'''
    
    file_paths = create_file_paths(set_size,npdf=npdf,data_size = data_size,path_to_directory=path_to_directory)
    
    for i,file in enumerate(file_paths):
        print(i)
        position = i*set_size
        save_set(file,position,set_size,npdf=npdf)
