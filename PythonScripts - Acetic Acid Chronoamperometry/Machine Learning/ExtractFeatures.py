import numpy as np
import pandas as pd
from pandas.io.stata import stata_epoch
from helper import find_csv,find_sigma,finddir


num_of_conc = 4

"""
def extractFeatures(CVs):
    features = pd.DataFrame(columns=['keq','kf','cbulk','steady state flux','reserved'])
    for index,CV in enumerate(CVs):
        sigma,keq,kf,k0,alpha,cbulk = find_sigma(CV)

        try:
            df = pd.read_csv(f'{path_to_training}/{CV}',header=None)
            if len(df) > 300:
                stedy_state_flux = (df.iloc[1,-1]* 2*np.pi*5e-6*2/np.pi*96485*9.311e-9*1000).mean()
                features = features.append(pd.Series([keq,kf,cbulk,stedy_state_flux,0.0],index=['keq','kf','cbulk','steady state flux','reserved']),ignore_index=True)
        except Exception as e:
            print(e)
            continue
 


    features['log10keq'] = np.log10(features['keq'])
    features['log10kf'] = np.log10(features['kf'])
    features.to_csv('features.csv',index=False)
"""
def extractFeatures(path_to_dir,CVs,columns):
    different_conc_features = list()
    for index,CV in enumerate(CVs):
        if index < num_of_conc:
            sigma,keq,kf,k0,alpha,cbulk = find_sigma(CV)
            try:
                df = pd.read_csv(f'{path_to_dir}/{CV}',header=None)
            except:
                return
            if len(df)<778:
                return
            stedy_state_flux = (df.iloc[-10:,1]* 2*np.pi*5e-6*2/np.pi*96485*9.311e-9*1000).mean()
            if index == 0:
                different_conc_features += [keq,kf]
            different_conc_features+= [cbulk,stedy_state_flux]
    
    
    return pd.Series(different_conc_features,index=columns)






def extracSeriesFeatures(basedir):
    columns = ['keq','kf'] 
    for i in range(num_of_conc):
        columns += [f'cbulk {i}',f'steady state flux {i}']
    features = pd.DataFrame(columns=columns)
    dirs = finddir(basedir)
    for directory in dirs[1:]:
        CVs = find_csv(directory)
        if len(CVs) == num_of_conc:
            different_conc_features = extractFeatures(directory,CVs,columns)

            features = features.append(different_conc_features,ignore_index=True)

    
    features['log10keq'] = np.log10(features['keq'])
    features['log10kf'] = np.log10(features['kf'])
    features.to_csv('features.csv',index=False)
 



if __name__ == "__main__":
    extracSeriesFeatures('../Series Exp T')
