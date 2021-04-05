from lifelines import KaplanMeierFitter
import pandas as pd
import matplotlib.pyplot as plt
## Example Data 
# durations = [5,6,6,2.5,4,4]
# event_observed = [1, 0, 0, 1, 1, 1]
df = pd.read_csv("data/kkbox_sample_small.csv")
durations = df[['duration']]
event_observed = df[['event']]


# Show plot 
# create a kmf object
kmf = KaplanMeierFitter() 

## Fit the data into the model
kmf.fit(durations, event_observed,label='Kaplan Meier Estimate')

## Create an estimate
kmf.plot(ci_show=False) ## ci_show is meant for Confidence interval, since our data set is too tiny, thus i am not showing it.
plt.show()