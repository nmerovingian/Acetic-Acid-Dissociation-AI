from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd 

num_of_conc = 4
degree = 3
def read_features(file_name='features.csv'):
    data = pd.read_csv(file_name)
    data = data.fillna(0.0)
    

    features_name = list() 
    for i in range(num_of_conc):
        features_name += [f'cbulk {i}',f'steady state flux {i}']
    features = data[features_name]
    targets = data[['log10kf','log10keq']]

    return features, targets

if __name__ == '__main__':
    features,targets = read_features()
    data_train,data_test,target_train,target_test = train_test_split(features,targets,test_size=0.1)
    scaler = MinMaxScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg.fit(data_train,target_train)


    preds = polyreg.predict(data_test)

    df = pd.DataFrame(preds, columns =['log10kf predicted','log10keq predicted'])
    df = pd.concat([df,target_test.reset_index(drop=True)],axis=1)
    df.to_csv(f'test_result {degree} polynomial.csv',index=False)
