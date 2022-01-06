from matplotlib import pyplot as plt
from matplotlib import cm,colors
from mpl_toolkits.mplot3d import Axes3D
from helper import format_func_dimensionless_keq,format_func_dimensionless_kf,format_func_pct,format_func_nA
from scipy.interpolate import interp2d
import numpy as np
import pandas as pd


linewidth = 3
fontsize = 14

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

df = pd.read_csv('features.csv')

conc_dic = {'steady state flux 0':0.1,'steady state flux 1':0.01,'steady state flux 2':0.02,'steady state flux 3':0.04}


for key,value in conc_dic.items():
        
        #fig = plt.figure(figsize=(8,4.5))
        fig = plt.figure(figsize=(16,9))
        fig.tight_layout()
        ax = fig.add_subplot(projection='3d')


        cmap = cm.viridis
        CS = ax.plot_trisurf(df['log10kf'],df['log10keq'],df[key],cmap=cmap,alpha=0.8)

        ax.set_xlabel(r'$log_{{10}}(K_f,s^{-1})$',fontsize='large',fontweight='bold',labelpad=10)
        ax.set_ylabel(r'$log_{{10}}(\kappa_{{eq}},M)$',fontsize='large',fontweight='bold',labelpad=10)
        #ax.set_zlabel(r'Predicted Steady State Current, nA',fontsize='large',fontweight='bold',labelpad=10)
        ax.zaxis.set_major_formatter(format_func_nA)
        cbax = fig.add_axes([0.08,0.1,0.02,0.8])
        fig.colorbar(CS,label=r'Predicted Steady State Current, nA',shrink=0.75,cax=cbax,format=format_func_nA)

        #f = interp2d(df['log10kf'],df['log10keq'],df[key],kind='cubic')
        """for xs in [5.89,6.20,6.54]:
                for ys in [-4.756]:
                        zpred = f(xs,ys)
                        if key == 'steady state flux 2':
                                print(zpred)
                        ax.scatter(xs,ys,zpred,marker='X',label=f'$log_{{10}}(k_f,s^{{-1}})={xs:.2f} log_{{10}}(\kappa_{{eq}},M )={ys:.2f}$',lw=6)
        ax.legend()
        """
        ax.view_init(45, 45)  #Elevation Azimuth
        ax.set_zlim(ax.get_zlim()[0]-2e-9,ax.get_zlim()[1])


        contourax = fig.add_axes([0.18,0.1,0.15,0.2])
        contourax.tricontourf(df['log10kf'],df['log10keq'],df[key],cmap=cmap,levels = 20,alpha=0.8)
        contourax.invert_xaxis()
        contourax.invert_yaxis()
        contourax.set_xlabel(r'$log_{{10}}(K_f,s^{-1})$',fontsize='medium',fontweight='bold',labelpad=10)
        contourax.set_ylabel(r'$log_{{10}}(\kappa_{{eq}},M)$',fontsize='medium',fontweight='bold',labelpad=10)
        fig.savefig(f'Predicted Current {conc_dic[key]:.2f}M HAc .png',dpi=250,bbox_inches = 'tight')



for key,value in conc_dic.items():

        fig,ax = plt.subplots(figsize=(16,9))



        cmap = cm.viridis
        CS = ax.tricontourf(df['log10kf'],df['log10keq'],df[key],cmap=cmap,alpha=0.8,levels=80)
        ax.set_xlabel(r'$log_{{10}}(K_f,s^{-1})$',fontsize='large',fontweight='bold',labelpad=10)
        ax.set_ylabel(r'$log_{{10}}(\kappa_{{eq}},M)$',fontsize='large',fontweight='bold',labelpad=10)
        #ax.set_zlabel(r'Predicted Steady State Current, nA',fontsize='large',fontweight='bold',labelpad=10)
        cbax = fig.add_axes([0.0,0.02,0.02,0.8])
        fig.colorbar(CS,label=r'Predicted Steady State Current, nA',shrink=0.75,cax=cbax,format=format_func_nA)


        fig.savefig(f'Contour Predicted Current {conc_dic[key]:.2f}M HAc .png',dpi=250,bbox_inches = 'tight')