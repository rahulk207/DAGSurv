import torch
import os
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from models import *
from utils import *
import pickle

def train(batch_size, X_train, E_train, Y_train, mask1_train):

    X, events, Y, temp_mask1 = f_get_minibatch(batch_size, X_train, E_train, Y_train, mask1_train)

    optimizer.zero_grad()
    out, adj_A, mu_z, logvar_z = net(X, Y.float())
    loss = criterion(out, events, temp_mask1, mu_z, logvar_z)
    l = loss.item()
    loss.backward()
    loss = optimizer.step()

    return l, adj_A

#data loading
#synthetic
"""dummy = pd.read_csv("data/dummy.csv")

label = np.asarray(dummy['Event']); label = label.reshape((len(label),1))
time = np.asarray(dummy['Time']); time = time.reshape((len(time),1))
X = np.asarray(dummy[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]])"""

#metabric
dummy = pd.read_csv("data/metabric.csv")

label = np.asarray(dummy['label']); label = label.reshape((len(label),1))
time = np.asarray(dummy['time']).astype(int); time = time.reshape((len(time),1))
# # # print(time)
X = np.asarray(dummy[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]])

#support
"""dummy = pd.read_csv("data/support.csv")

label = np.asarray(dummy['label']); label = label.reshape((len(label),1))
time = np.asarray(dummy['time']); time = time.reshape((len(time),1))
# print(time)
X = np.asarray(dummy[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14"]])"""

# gbsg
# dummy = pd.read_csv("data/gbsg.csv")

# label = np.asarray(dummy['label']); label = label.reshape((len(label),1))
# time = np.asarray(dummy['time']).astype(int); time = time.reshape((len(time),1))
# # print(time)
# X = np.asarray(dummy[["x1", "x2", "x3", "x4", "x5", "x6", "x7"]])

"""dummy = pd.read_csv("synthetic_final.csv")

label = np.asarray(dummy['label']); label = label.reshape((len(label),1))
time = np.asarray(dummy['time']).astype(int); time = time.reshape((len(time),1))
# print(time)
X = np.asarray(dummy[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]])"""
# X = f_get_Normalization(X, 'standard')

num_category = int(np.max(time) * 1.2)
mask1 = create_mask1(time,label,num_category)

# time = f_get_Normalization(time, 'standard')
X_train,X_test,Y_train,Y_test, E_train,E_test, mask1_train, mask1_test = train_test_split(X, time, label, mask1, test_size=0.2, random_state=1234)


X_train,X_val,Y_train,Y_val, E_train,E_val, mask1_train, mask1_val = train_test_split(X_train, Y_train, E_train, mask1_train, test_size=0.2, random_state=1234)


#main
OUT_ITR = 50
dt_now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

for itr in range(OUT_ITR):
    print("itreration "+str(itr+1)+" started!")
    params = get_random_hyperparameters(n_in_x = 9, n_in_y = 1, n_out = 10)
    # params = {'batch_size': 128, 'EPOCHS': 65000, 'lr': 1e-05, 'n_hid1': 64, 'n_hid2': 64, 'num_layers1': 2, 'num_layers2': 2, 'active_fn': 'ELU', 'alpha': 1.0, 'beta': 1.0, 'n_in_x': 9, 'n_in_y': 1, 'n_out': 10}
    # params = {
    #             'EPOCHS' : 50000,
    #             'lr'     : 1e-5,
    #             'n_in_x' : 10,
    #             'n_in_y' : 1,
    #             'n_hid1' : 128,
    #             'n_hid2' : 32,
    #             'n_out'  : 11,
    #             'alpha'  : 1,
    #             'beta'   : 1,
    #             'batch_size' : 64
    #             #activation-fn
    # }

    net = modCVAE(params['n_in_x'], params['n_in_y'], num_category, params['n_hid1'], params['n_hid2'], params['n_out'], params['num_layers1'], params['num_layers2'], params['active_fn'])
    criterion = CVAELoss(params['alpha'], params['beta'])
    optimizer = optim.Adam(net.parameters(), params['lr'])

    num_Event = 1

    flag = False

    epoch_index1, epoch_index2, losses, c_index = [], [], [], []
    for i in range(params['EPOCHS']):
        net.train()
        # print("EPOCH:", i)
        loss, adj_A = train(params['batch_size'], X_train, E_train, Y_train, mask1_train)

        if loss != loss: #NaN
            flag = True
            print('HEY')
            break

        epoch_index1.append(i); losses.append(min(10, loss))
        # print("Epoch",i+1,loss)
        if i % 1000 == 1:
            net.eval()
            with torch.no_grad():
                X_val = torch.Tensor(X_val).float()

                Z_val = torch.randn(list(X_val.size())[0], params['n_in_x']+params['n_in_y'])
                input = torch.cat([Z_val, X_val], 1)
                hr_pred2 = net.decoder(input, adj_A)
                hr_pred2 = hr_pred2.reshape((list(hr_pred2.size())[0], num_Event, list(hr_pred2.size())[1]))
                hr_pred2 = hr_pred2.detach().numpy()

                EVAL_TIMES = [12, 24, 36]
                FINAL1 = np.zeros([num_Event, len(EVAL_TIMES), 1])
                result1 = np.zeros([num_Event, len(EVAL_TIMES)])
                for t, t_time in enumerate(EVAL_TIMES):
                    eval_horizon = int(t_time)

                    if eval_horizon >= num_category:
                        print( 'ERROR: evaluation horizon is out of range')
                        result1[:, t] = -1
                    else:
                        # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
                        risk = np.sum(hr_pred2[:,:,:(eval_horizon+1)], axis=2) #risk score until EVAL_TIMES
                        for k in range(num_Event):
                            # result1[k, t] = c_index(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                            # result2[k, t] = brier_score(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                            result1[k, t] = weighted_c_index(Y_train, (E_train[:,0] == k+1).astype(int), risk[:,k], Y_val, (E_val[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable))
                tmp_valid = np.mean(result1)
                epoch_index2.append(i); c_index.append(tmp_valid)
                # print("C-Index :", tmp_valid)

    if flag:
        continue

    my_path = os.getcwd()
    my_file = 'results/'+dt_now+'/iter '+str(itr+1)
    path = os.path.join(my_path, my_file)
    os.makedirs( path, 0o777)
    plt.plot(epoch_index1, losses)
    plt.savefig('results/'+dt_now+'/iter '+str(itr+1)+'/Loss_Plot.png')
    plt.close()

    plt.plot(epoch_index2, c_index)
    plt.savefig('results/'+dt_now+'/iter '+str(itr+1)+'/c_INDEX_Plot.png')
    plt.close()

    fp = open('results/'+dt_now+'/iter '+str(itr+1)+"/logs.txt", "a")
    fp.write("METABRIC learnable ")
    fp.write(str(params))
    fp.write("Train C-Index :"+str(max(c_index)))

    f = open('results/'+dt_now+'/iter '+str(itr+1)+'/model', "wb")
    pickle.dump(net, f)
    #test()
    print(adj_A)
    net.eval()
    with torch.no_grad():

        f = open('results/'+dt_now+'/iter '+str(itr+1)+'/model', "rb")
        net = pickle.load(f)

        X_te = torch.Tensor(X_test).float()

        Z_te = torch.randn(list(X_te.size())[0], params['n_in_x']+params['n_in_y'])
        input = torch.cat([Z_te, X_te], 1)
        hr_pred2 = net.decoder(input, adj_A)
        hr_pred2 = hr_pred2.reshape((list(hr_pred2.size())[0], num_Event, list(hr_pred2.size())[1]))
        hr_pred2 = hr_pred2.detach().numpy()

        EVAL_TIMES = [12, 24, 36]
        FINAL1 = np.zeros([num_Event, len(EVAL_TIMES), 1])
        result1 = np.zeros([num_Event, len(EVAL_TIMES)])
        for t, t_time in enumerate(EVAL_TIMES):
            eval_horizon = int(t_time)

            if eval_horizon >= num_category:
                print( 'ERROR: evaluation horizon is out of range')
                result1[:, t] = -1
            else:
                # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
                risk = np.sum(hr_pred2[:,:,:(eval_horizon+1)], axis=2) #risk score until EVAL_TIMES
                for k in range(num_Event):
                    # result1[k, t] = c_index(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                    # result2[k, t] = brier_score(risk[:,k], te_time, (te_label[:,0] == k+1).astype(float), eval_horizon) #-1 for no event (not comparable)
                    result1[k, t] = weighted_c_index(Y_train, (E_train[:,0] == k+1).astype(int), risk[:,k], Y_test, (E_test[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

        tmp_valid = np.mean(result1)
        fp.write("Test C-Index :"+str(tmp_valid))
    fp.close()
    print("itreration "+str(itr+1)+" complete!")
