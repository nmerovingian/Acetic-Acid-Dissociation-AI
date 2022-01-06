import tensorflow
tensorflow.random.set_seed(0) # random seeds for reproducibility
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense,Input,Concatenate
#from tensorflow.keras.layers import concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
np.random.seed(0) # random seeds for reproducibility
import os

epochs = 10000 # we used 10k eqpochs                     
num_of_conc = 4
def read_features(file_name='features.csv'):
    data = pd.read_csv(file_name)
    data = data.fillna(0.0)
    

    features_name = list() 
    for i in range(num_of_conc):
        features_name += [f'cbulk {i}',f'steady state flux {i}']
    features = data[features_name]
    targets = data[['log10kf','log10keq']]

    return features, targets



def create_model(data=None,output_shape=0,optimizer='Adam',loss='mean_absolute_error'):
    # head1 
    inputs1 = Input(shape=(data.shape[1],))
    dnn11 = Dense(32,activation='relu')(inputs1)
    dnn12 = Dense(16,activation='relu')(dnn11)
    dnn13 = Dense(12,activation='relu')(dnn12)
    dnn14 = Dense(6,activation='linear')(dnn13)
    #head2
    inputs2 = Input(shape=(data.shape[1],))
    dnn21 = Dense(32,activation='relu')(inputs2)
    dnn22 = Dense(16,activation='relu')(dnn21)
    dnn23 = Dense(12,activation='relu')(dnn22)
    dnn24 = Dense(6,activation='linear')(dnn23)

    # merge
    merged = Concatenate()([dnn14,dnn24])
    outputs = Dense(output_shape)(merged)

    model = Model(inputs=[inputs1,inputs2],outputs=outputs)
    model.compile(optimizer=optimizer,loss=loss,metrics=['mean_squared_error','mean_absolute_error'])
    return model

def schedule(epoch,lr):
    if epoch <3000:
        return lr
    else:
        #return lr*0.9999
        return lr
if __name__ == '__main__':
    features,targets = read_features()
    data_train,data_test,target_train,target_test = train_test_split(features,targets,test_size=0.1)
    scaler = MinMaxScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    scheduler = LearningRateScheduler(schedule,verbose=0)


    model = create_model(data=data_train,output_shape=targets.shape[1])

    if not os.path.exists(f'./weights/Predict kf and keq {epochs}.h5'):
        history = model.fit([data_train,data_train],target_train,epochs=epochs,batch_size=32,validation_split=0.2,verbose=1,callbacks=[scheduler])
        model.save_weights(f'./weights/Predict kf and keq {epochs}.h5')

        df = pd.DataFrame(history.history)
        df.iloc[100:].plot()
        plt.savefig('Training History.png')
    else:
        print('Using existing weights')
        model.load_weights(f'./weights/Predict kf and keq {epochs}.h5')

    preds = model.predict([data_test,data_test])

    df = pd.DataFrame(preds, columns =['log10kf predicted','log10keq predicted'])
    df = pd.concat([df,target_test.reset_index(drop=True)],axis=1)
    df.to_csv('test_result.csv',index=False)
   
    
    experimental = pd.read_csv('Experimental Results.csv')
    experimental_feature  = scaler.transform(experimental[['cbulk 0','steady state flux 0','cbulk 1','steady state flux 1','cbulk 2','steady state flux 2','cbulk 3','steady state flux 3']])
    experimental_preds = model.predict([experimental_feature,experimental_feature])
    experimental['log10kf predicted'] = experimental_preds[:,0]
    experimental['log10keq predicted'] = experimental_preds[:,1]
    experimental.to_csv(f'Experimental prediction {epochs}.csv',index=False)





