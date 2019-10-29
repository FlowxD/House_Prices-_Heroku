# -*- coding: utf-8 -
"""
Created on Mon Oct 28 14:17:52 2019

@author: Mandar Joshi
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    int_features = [21132,1231,4121,1214,123,412,12414,512,115,155]
    int_features = [x for x in request.form.values()]

    from scipy.special import boxcox1p
    lam = 0.15
    int_features[0] = np.int64(int_features[0])

    int_features[0] = boxcox1p(int_features[0], lam)
  
    int_features[1] = np.int64(int_features[1])

    int_features[1] = boxcox1p(int_features[1], lam)
    
    int_features[2] = np.int64(int_features[2])

    int_features[2] = boxcox1p(int_features[2], lam)
    
    int_features[3] = np.int64(int_features[3])

    int_features[3] = boxcox1p(int_features[3], lam)

    int_features[4] = np.int64(int_features[4])

    int_features[4] = boxcox1p(int_features[4], lam)

    int_features[5] = np.int64(int_features[5])

    int_features[5] = boxcox1p(int_features[5], lam)

    int_features[6] = np.int64(int_features[6])

    int_features[6] = boxcox1p(int_features[6], lam)

    int_features[7] = np.int64(int_features[7])

    int_features[7] = boxcox1p(int_features[7], lam)

    int_features[8] = np.int64(int_features[8])

    int_features[8] = boxcox1p(int_features[8], lam)
        
    int_features[9] = np.int64(int_features[9])

    int_features[9] = boxcox1p(int_features[9], lam)
    
    from scipy.special import inv_boxcox1p
        
    columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'TotalSF']
    values = [7.990963041593332, 1.8203341036428238, 9.125735246126716, 35.391370879389704,0.7304631471189666, 0.7304631471189666, 1.5409627556327752, 1.5409627556327752, 0.0005003172240540867, 1.4013368444274956, 0.0431549546952863, 2.9929161754443663, int_features[5], 1.1948967456559554, 0.24556737057378078, 1.4155057023841355, int_features[1], 2.1604465664352563, 14.13734536216387, int_features[7], 0.8932636402684723, 0.7538300024576454, 2.7582173064497746, 2.844430978246579, 1.071057748915512, 3.268986443449214, 1.3624422931139548, 1.7328990595033906, 0.8834994399659468, 1.2470924388496651, 1.4569085852270438, 1.2062523036053971, 1.317433970123405, 7.084892889342716, 1.9657590475261213, 1.0517430422748202, 9.391255388699905, 11.89254623970312, 0.7446911405592987, 0.7563474000353817, 0.6829330108338099, 1.6815016774004494, int_features[4], 4.917140353780343, 0.15794652317139848, int_features[2], 0.30751236837651724, 0.04166142948306093, int_features[6], 0.2774860242213367, 1.4742074483529004, 0.7517241594265581, 1.2746341259746379,int_features[8] , 2.1887574652310366, 0.42455642997342163, 1.3031568396814526, 1.140331159374495, 14.151202776364704, 0.7722995531965653, int_features[3],int_features[9], 1.774185368855956, 1.7865149945239338, 1.1111640610733957, 3.726756270282964, 3.253621458595589, 1.0382729368834795, 0.13072845037668754, 0.6225990043864975, 0.050872777658061384, 1.1920461692064572, 1.131509564697584, 1.1922835973693042, 0.40283910651921423, 2.230572790696048, 14.195035682494952, 2.478799472550466, 1.7025947724500126, int_features[0]]
    xdd=dict(zip(columns, values))

    df_s = pd.DataFrame(columns=columns)
    
    df_s = df_s.append(xdd, ignore_index=True)
    prediction = model.predict(df_s)

    
    print(prediction)
    
    prediction = inv_boxcox1p(prediction, 0.15)
#addiiton to look realistic 
    prediction = inv_boxcox1p(prediction, 0.15)
    
    print(prediction)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='House price should be $ {}'.format(output))
 

if __name__ == "__main__":  
    app.run(debug=True)
