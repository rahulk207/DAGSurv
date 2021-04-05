import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
import torch
import torchtuples as tt

from pycox.datasets import gbsg
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

import pickle

np.random.seed(1234)
_ = torch.manual_seed(1234)
dummy = gbsg.read_df()
df_train, df_test = train_test_split(dummy, test_size=0.2, random_state=1234)
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=1234)

cols_standardize = ['x3', 'x4', 'x5', 'x6']
cols_leave = ['x0', 'x1', 'x2']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val, y_val

in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)

model = CoxPH(net, tt.optim.Adam)
batch_size = 256
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
_ = lrfinder.plot()
# print(lrfinder.get_best_lr())
model.optimizer.set_lr(0.01)

epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val, val_batch_size=batch_size)
print(model.partial_log_likelihood(*val).mean())

_ = model.compute_baseline_hazards()

# f = open('GBSGModel', 'wb')
# pickle.dump(model, f)
# f.close()

bootstrap_ci = []
for i in range(list(x_test.shape)[0]):
	c = np.random.choice(list(x_test.shape)[0], list(x_test.shape)[0])
	x_temp = x_test[c]; durations_temp = durations_test[c]; events_temp = events_test[c]

	surv = model.predict_surv_df(x_temp)
	ev = EvalSurv(surv, durations_temp, events_temp, censor_surv='km')
	tmp_valid = ev.concordance_td(); print(i, tmp_valid)
	bootstrap_ci.append(tmp_valid)

np.save("bootstrap_ci_deepSurv_synthetic", np.array(bootstrap_ci))
print(np.mean(np.array(bootstrap_ci)))



