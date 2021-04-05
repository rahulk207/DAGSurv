from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from utils import *

np.random.seed(1234)
torch.manual_seed(1234)
f = open('model_kkbox', "rb")
net = pickle.load(f)

dummy = pd.read_csv("data/kkbox_sample.csv")

label = np.asarray(dummy['event']); label = label.reshape((len(label),1))
time = np.asarray(dummy['duration']).astype(int); time = time.reshape((len(time),1))
# print(time)
X = np.asarray(dummy[["n_prev_churns", "log_days_between_subs", "log_days_since_reg_init", "payment_method_id", "log_payment_plan_days", "log_plan_list_price", "log_actual_amount_paid", "is_auto_renew", "is_cancel", "city", "gender", "registered_via", "age_at_start", "strange_age", "nan_days_since_reg_init", "no_prev_churns"]])


# dummy = pd.read_csv("data/gbsg.csv")

# label = np.asarray(dummy['label']); label = label.reshape((len(label),1))
# time = np.asarray(dummy['time']).astype(int); time = time.reshape((len(time),1))
# # print(time)
# X = np.asarray(dummy[["x1", "x2", "x3", "x4", "x5", "x6", "x7"]])

# dummy = pd.read_csv("data/metabric.csv")

# label = np.asarray(dummy['label']); label = label.reshape((len(label),1))
# time = np.asarray(dummy['time']).astype(int); time = time.reshape((len(time),1))
# # # # print(time)
# X = np.asarray(dummy[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]])

# X = f_get_Normalization(X, 'standard')

num_category = int(np.max(time) * 1.2)
num_Event = 1

X_train,X_test,Y_train,Y_test, E_train,E_test = train_test_split(X, time, label, test_size=0.2, random_state=1234)
X_train,X_val,Y_train,Y_val, E_train,E_val = train_test_split(X_train, Y_train, E_train, test_size=0.2, random_state=1234)


adj_A = torch.from_numpy(np.load("graphs/kkbox_graph.npy"))
X_te = torch.Tensor(X_test).float()

Z_te = torch.randn(list(X_te.size())[0], 17)
bootstrap_ci = []
for i in range(list(X_te.size())[0]):
	c = np.random.choice(list(X_te.size())[0], list(X_te.size())[0])
	X_temp = X_te[c]; Y_temp = Y_test[c]; E_temp = E_test[c]
	input = torch.cat([Z_te, X_temp], 1)
	hr_pred2 = net.decoder(input, adj_A)
	hr_pred2 = hr_pred2.reshape((list(hr_pred2.size())[0], num_Event, list(hr_pred2.size())[1]))
	hr_pred2 = hr_pred2.detach().numpy()

	EVAL_TIMES = [50, 100, 150]
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
				result1[k, t] = weighted_c_index(Y_train, (E_train[:,0] == k+1).astype(int), risk[:,k], Y_temp, (E_temp[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)

	tmp_valid = np.mean(result1); print(i, tmp_valid)
	bootstrap_ci.append(tmp_valid)

np.save("bootstrap_ci_HG-SURV_kkbox", np.array(bootstrap_ci))
print(np.mean(np.array(bootstrap_ci)))

