import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
import torch
import torchtuples as tt
import pandas as pd
from pycox.models import CoxTime
from pycox.evaluation import EvalSurv
from pycox.models.cox_time import MLPVanillaCoxTime

import pickle

np.random.seed(1234)
_ = torch.manual_seed(1234)

dummy = pd.read_csv("kkbox_sample.csv")
df_train, df_test = train_test_split(dummy, test_size=0.2, random_state=1234)
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=1234)

cols_standardize = ["n_prev_churns", "log_days_between_subs", "log_days_since_reg_init", "payment_method_id", "log_payment_plan_days", "log_plan_list_price", "log_actual_amount_paid", "age_at_start"]
cols_leave = ["is_auto_renew", "is_cancel", "city", "gender", "registered_via", "strange_age", "nan_days_since_reg_init", "no_prev_churns"]

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

labtrans = CoxTime.label_transform()
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))
durations_test, events_test = get_target(df_test)
val = tt.tuplefy(x_val, y_val)

val.shapes()
val.repeat(2).cat().shapes()

in_features = x_train.shape[1]
num_nodes = [32, 32]
batch_norm = True
dropout = 0.1
net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)

model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

batch_size = 256
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=2)
_ = lrfinder.plot()

model.optimizer.set_lr(0.01)

epochs = 512
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

print(y_train)
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,val_data=val.repeat(10).cat())
print(model.partial_log_likelihood(*val).mean())

_ = model.compute_baseline_hazards()

bootstrap_ci = []
for i in range(list(x_test.shape)[0]):
	c = np.random.choice(list(x_test.shape)[0], list(x_test.shape)[0])
	x_temp = x_test[c]; durations_temp = durations_test[c]; events_temp = events_test[c]

	surv = model.predict_surv_df(x_temp)
	ev = EvalSurv(surv, durations_temp, events_temp, censor_surv='km')
	tmp_valid = ev.concordance_td(); print(i, tmp_valid)
	bootstrap_ci.append(tmp_valid)

np.save("bootstrap_ci_cox_kkbox", np.array(bootstrap_ci))
print(np.mean(np.array(bootstrap_ci)))
