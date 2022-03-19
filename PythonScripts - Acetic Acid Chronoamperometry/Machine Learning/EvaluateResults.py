import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from helper import format_func_pct

linewidth = 3
fontsize = 14
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

error_threshold = 0.1

df = pd.read_csv('test_result.csv')

#df = np.power(10,df)
df = df[df['log10kf']>0]
df['kf prediction error'] = (df['log10kf predicted'] - df['log10kf'])/df['log10kf']
df = df[df['kf prediction error'].abs()<0.5]
df['keq prediction error'] = (df['log10keq predicted'] -df['log10keq'])/df['log10keq']

fig,axes = plt.subplots(2,1,figsize=(8,6))
fig.subplots_adjust(hspace=0.4)
ax = axes[0]
sns.histplot(data=df['kf prediction error'],bins=200,binrange=(-0.5,0.5),kde=True,stat='probability',ax=ax)
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Error of predicting $log_{{10}}k_{{f}}$',fontsize='large',fontweight='bold')
ax.xaxis.set_major_formatter(format_func_pct)
mean_error = df['kf prediction error'].mean()
std_error = df['kf prediction error'].std()
error_within_pct = len(df[df['kf prediction error'].abs()<error_threshold])/(len(df))
# estimate stdev of yhat
sum_errs = df['kf prediction error'].std()
stdev = np.sqrt(1.0/(len(df)-2)*sum_errs)
# interval
interval = 1.96*stdev
print('interval of logkf prediction',interval)
#ax.set_title(f'Error of predicting log10kf.\n{error_within_pct:.1%} of predictions are within {error_threshold:.0%} error')

error_threshold = 0.05
ax = axes[1]
sns.histplot(data=df['keq prediction error'],bins=200,binrange=(-0.5,0.5),kde=True,stat='probability',ax=ax)
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Error of predicting $log_{{10}}K_{{eq}}$',fontsize='large',fontweight='bold')
ax.xaxis.set_major_formatter(format_func_pct)
mean_error = df['keq prediction error'].mean()
std_error = df['keq prediction error'].std()
error_within_pct = len(df[df['keq prediction error'].abs()<error_threshold])/(len(df))
#ax.set_title(f'Error of predicting log10keq.\n{error_within_pct:.1%} of predictions are within {error_threshold:.0%} error')
# estimate stdev of yhat
sum_errs = df['keq prediction error'].std()
stdev = np.sqrt(1.0/(len(df)-2)*sum_errs)
# interval
interval = 1.96*stdev
print('interval of logkf prediction',interval)

fig.text(0.13,0.82,'(a)',fontsize=20,fontweight='bold')
fig.text(0.13,0.37,'(b)',fontsize=20,fontweight='bold')

fig.savefig('Test result.png',dpi=250,bbox_inches='tight')


error_threshold = 0.1

degree = 3
df = pd.read_csv(f'test_result {degree} polynomial.csv')

#df = np.power(10,df)
df = df[df['log10kf']>0]
df['kf prediction error'] = (df['log10kf predicted'] - df['log10kf'])/df['log10kf']
df = df[df['kf prediction error'].abs()<0.5]
df['keq prediction error'] = (df['log10keq predicted'] -df['log10keq'])/df['log10keq']

fig,axes = plt.subplots(2,1,figsize=(8,6))
fig.subplots_adjust(hspace=0.4)
ax = axes[0]
sns.histplot(data=df['kf prediction error'],bins=200,binrange=(-0.5,0.5),kde=True,stat='probability',ax=ax)
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Error of predicting $log_{{10}}k_{{f}}$',fontsize='large',fontweight='bold')
ax.xaxis.set_major_formatter(format_func_pct)
mean_error = df['kf prediction error'].mean()
std_error = df['kf prediction error'].std()
error_within_pct = len(df[df['kf prediction error'].abs()<error_threshold])/(len(df))
# estimate stdev of yhat
sum_errs = df['kf prediction error'].std()
stdev = np.sqrt(1.0/(len(df)-2)*sum_errs)
# interval
interval = 1.96*stdev
print('interval of logkf prediction',interval)
#ax.set_title(f'Error of predicting log10kf.\n{error_within_pct:.1%} of predictions are within {error_threshold:.0%} error')

error_threshold = 0.05
ax = axes[1]
sns.histplot(data=df['keq prediction error'],bins=200,binrange=(-0.5,0.5),kde=True,stat='probability',ax=ax)
ax.set_xlim(-0.5,0.5)
ax.set_xlabel(r'Error of predicting $log_{{10}}K_{{eq}}$',fontsize='large',fontweight='bold')
ax.xaxis.set_major_formatter(format_func_pct)
mean_error = df['keq prediction error'].mean()
std_error = df['keq prediction error'].std()
error_within_pct = len(df[df['keq prediction error'].abs()<error_threshold])/(len(df))
#ax.set_title(f'Error of predicting log10keq.\n{error_within_pct:.1%} of predictions are within {error_threshold:.0%} error')
# estimate stdev of yhat
sum_errs = df['keq prediction error'].std()
stdev = np.sqrt(1.0/(len(df)-2)*sum_errs)
# interval
interval = 1.96*stdev
print('interval of logkf prediction',interval)

fig.text(0.13,0.82,'(a)',fontsize=20,fontweight='bold')
fig.text(0.13,0.37,'(b)',fontsize=20,fontweight='bold')

fig.savefig(f'Test result {degree} polynomial.png',dpi=250,bbox_inches='tight')