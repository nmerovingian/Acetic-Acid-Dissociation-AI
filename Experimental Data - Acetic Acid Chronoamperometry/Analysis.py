from datetime import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
txts = ['10 mM.txt','20 mM.txt','40 mM.txt','100 mM.txt']
import numpy as np
from matplotlib import cm 
colors = cm.viridis(np.linspace(0,1,len(txts)))
index = 0

linewidth = 3
fontsize = 14
figsize = [10,8]

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs







def analysis(time_limit):
    fig,ax = plt.subplots(figsize=(16,9))
    for index, txt in enumerate(txts):
        current_trials = []
        current_trials_std = []
        scan_rates = np.zeros((len(txts),1))
        forward_scan_peak_currents = np.zeros((len(txts)))
        forward_scan_peak_potentials  = np.zeros((len(txts)))
        df = pd.read_csv(txt,delimiter=',')
        df = df[df.iloc[:,0]<time_limit]




        df_forward = df.copy()

        peak_current = df_forward.iloc[-200:,1].mean()
        peak_current_std = df_forward.iloc[-200:,1].std()
        current_trials.append(peak_current)
        current_trials_std.append(peak_current_std)


        forward_scan_peak_currents[index] = df_forward.iloc[:,1].min()

        forward_scan_peak_potential = df_forward.iloc[df_forward.iloc[:,1].idxmin(),0]
        df_forward.iloc[:,1] *= 1e9
        df_forward.plot(x=0,y=1,label=f"{txt.replace('.txt','')}",ax=ax,color=tuple(colors[index]),linewidth=3)




    ax.set_xlim(-time_limit*0.1,time_limit+0.1)
    ax.set_xlabel('Time, s',fontsize='large')
    ax.set_ylabel('Current, nA',fontsize='large')
    #ax.set_title(f'Different cocentration of acetic acid in 100 mM $KNO_3$\nPt microdisc electrode, diameter = $10\mu m$')

    fig.savefig(f'KNO3 Electrolyte, Compare Concentrtaion at {time_limit:.2f}s.png',dpi=400,bbox_inches = 'tight')




if __name__ == '__main__':
    time_limits = [10]
    for time_limit in time_limits:
        analysis(time_limit)