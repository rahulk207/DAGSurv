import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

def confidenceInterval(data):
	median = np.median(data)
	upper_quartile = np.percentile(data, 75)
	lower_quartile = np.percentile(data, 25)

	iqr = upper_quartile - lower_quartile

	dev = 1.57*iqr/(data.shape[0]**(0.5))
	print("Confidence Interval of median: ", median,"+/-",dev)
	
	return median, dev

boot_hg = np.load("bootstrap_ci_HG-SURV_kkbox.npy")
print(confidenceInterval(boot_hg))
boot_deepSurv = np.load("bootstrap_ci_deepSurv_kkbox.npy")
print(confidenceInterval(boot_deepSurv))
boot_cox = np.load("bootstrap_ci_cox_kkbox.npy")
print(confidenceInterval(boot_cox))
boot_deepHit = np.load("bootstrap_ci_deepHit_kkbox.npy")
print(confidenceInterval(boot_deepHit))

data = [boot_hg, boot_deepHit, boot_deepSurv, boot_cox]

fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
colors = ['black', 'green', 'purple', 'royalblue']
bp = ax.boxplot(data, patch_artist=True, notch=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticklabels(['DAGSurv', 'DeepHit', 'DeepSurv', 'CoxTime'], fontsize=15)
plt.yticks(size = 15)
# Save the figure
fig.savefig('boxPlot_kkbox_ci.png', bbox_inches='tight')