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


df = df[df['log10kf']>0]
df['kf prediction error'] = (df['log10kf predicted'] - df['log10kf'])/df['log10kf']
df = df[df['kf prediction error'].abs()<0.5]
df['keq prediction error'] = (df['log10keq predicted'] -df['log10keq'])/df['log10keq']

fig,axes = plt.subplots(2,1,figsize=(16,9))
fig.subplots_adjust(hspace=0.4)
ax = axes[0]
sns.histplot(data=df['kf prediction error'],bins=200,binrange=(-0.5,0.5),kde=True,stat='probability',ax=ax)
ax.set_xlim(-0.5,0.5)
ax.set_xlabel('Error of predicting log10kf')
ax.xaxis.set_major_formatter(format_func_pct)
mean_error = df['kf prediction error'].mean()
std_error = df['kf prediction error'].std()
error_within_pct = len(df[df['kf prediction error'].abs()<error_threshold])/(len(df))
ax.set_title(f'Error of predicting log10kf.\n{error_within_pct:.1%} of predictions are within {error_threshold:.0%} error')


ax = axes[1]
sns.histplot(data=df['keq prediction error'],bins=200,binrange=(-0.5,0.5),kde=True,stat='probability',ax=ax)
ax.set_xlim(-0.5,0.5)
ax.set_xlabel('Error of predicting log10Keq')
ax.xaxis.set_major_formatter(format_func_pct)
mean_error = df['keq prediction error'].mean()
std_error = df['keq prediction error'].std()
error_within_pct = len(df[df['keq prediction error'].abs()<error_threshold])/(len(df))
ax.set_title(f'Error of predicting log10keq.\n{error_within_pct:.1%} of predictions are within {error_threshold:.0%} error')


fig.text(0.13,0.82,'(A)',fontsize=20,fontweight='bold')
fig.text(0.13,0.37,'(B)',fontsize=20,fontweight='bold')

fig.savefig('Test result.png',dpi=250,bbox_inches='tight')