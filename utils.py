import torch
import random
import numpy as np
from torch import nn
from lifelines import KaplanMeierFitter

class CVAELoss(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def loglikelihood(self, out, k, mask1):

        I_1 = torch.sign(k)
        # tmp1 = torch.sum(torch.sum(mask1 * out, 1), 0, keepdim = True)
        tmp1 = torch.sum(mask1 * out, 1, keepdim = True)
        tmp1 = I_1 * torch.log(tmp1 + 1e-08)

        # tmp2 = torch.sum(torch.sum(mask1 * out, 1), 0, keepdim = True)
        tmp2 = torch.sum(mask1 * out, 1, keepdim = True)
        tmp2 = (1. - I_1) * torch.log(tmp2 + 1e-08)

        return -torch.mean(tmp1 + 1.0*tmp2)

    def kl_gaussian(self, mu, logvar):
        #to refer
        return -0.5 * torch.sum(1 + logvar - mu*mu - torch.exp(logvar))

    def forward(self, out, k, mask1, mu, logvar):
        return (self.alpha*self.loglikelihood(out, k, mask1)) + (self.beta*self.kl_gaussian(mu, logvar))

### C(t)-INDEX CALCULATION: this account for the weighted average for unbaised estimation
def weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    G = CensoringProb(Y_train, T_train)

    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        tmp_idx = np.where(G[0,:] >= T_test[i])[0]

        if len(tmp_idx) == 0:
            W = (1./G[1, -1])**2
        else:
            W = (1./G[1, tmp_idx[0]])**2

        A[i, np.where(T_test[i] < T_test)] = 1. * W
        Q[i, np.where(Prediction[i] > Prediction)] = 1. # give weights

        if (T_test[i]<=Time and Y_test[i]==1):
            N_t[i,:] = 1.

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result

##### WEIGHTED C-INDEX & BRIER-SCORE
def CensoringProb(Y, T):

    T = T.reshape([-1]) # (N,) - np array
    Y = Y.reshape([-1]) # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y==0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  #fill 0 with ZoH (to prevent nan values)

    return G

def preprocess_adj_new(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
    return adj_normalized

def create_mask1(time, label, num_Category):
    mask1 = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            mask1[i, int(time[i,0])] = 1
        else: #label[i,2]==0: censored
            mask1[i,int(time[i,0]+1):] = 1
    return torch.from_numpy(mask1).float()

def f_get_minibatch(mb_size, x, label, time, mask1):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)
    x_mb = torch.from_numpy(x[idx, :]).float()
    k_mb = torch.from_numpy(label[idx, :]) # censoring(0)/event label(1)
    t_mb = torch.from_numpy(time[idx, :])
    m1_mb = mask1[idx, :].float() #fc_mask
    return x_mb, k_mb, t_mb, m1_mb

def f_get_Normalization(X, norm_mode):
    num_Patient, num_Feature = np.shape(X)

    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))/np.std(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.mean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j]) - np.min(X[:,j]))
    else:
        print("INPUT MODE ERROR!")

    return X

def get_random_hyperparameters(n_in_x, n_in_y, n_out):
    SET_BATCH_SIZE    = [32, 64, 128] #mb_size

    SET_LAYERS1        = [3, 4, 5] #number of layers

    SET_LAYERS2        = [3, 4, 5] #number of layers

    SET_NHID1         = [64, 128, 256] #number of nodes

    SET_NHID2         = [16, 32, 64]

    SET_ACTIVATION_FN = ['ReLU', 'ELU', 'SELU'] #non-linear activation functions

    SET_BETA          = [0.1, 0.5, 1.0, 3.0] #beta values -> ranking loss

    SET_LR            = [1e-4, 1e-5]

    SET_EPOCHS        = [100000, 150000]

    new_parser = {'batch_size': SET_BATCH_SIZE[np.random.randint(len(SET_BATCH_SIZE))],
                 'EPOCHS': SET_EPOCHS[np.random.randint(len(SET_EPOCHS))],
                 'lr': SET_LR[np.random.randint(len(SET_LR))],
                 'n_hid1': SET_NHID1[np.random.randint(len(SET_NHID1))],
                 'n_hid2': SET_NHID2[np.random.randint(len(SET_NHID2))],
                 'num_layers1':SET_LAYERS1[np.random.randint(len(SET_LAYERS1))],
                 'num_layers2':SET_LAYERS2[np.random.randint(len(SET_LAYERS2))],
                 'active_fn': SET_ACTIVATION_FN[np.random.randint(len(SET_ACTIVATION_FN))],
                 'alpha':1.0, #default (set alpha = 1.0 and change beta and gamma)
                 'beta':SET_BETA[np.random.randint(len(SET_BETA))],
                 'n_in_x' : n_in_x,
                 'n_in_y' : n_in_y,
                 'n_out'  : n_out
                 }

    return new_parser #outputs the dictionary of the randomly-chosen hyperparamters
