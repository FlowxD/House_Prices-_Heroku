# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:23:43 2019

@author: Mandar Joshi
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from sklearn.preprocessing import LabelEncoder  ###for encode a categorical values
from sklearn.model_selection import train_test_split  ## for spliting the data
from sklearn.preprocessing import StandardScaler

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.head()

miss_col_train = [col for col in train.columns if train[col].isnull().any()]
miss_col_test = [col for col in test.columns if test[col].isnull().any()]

for col in miss_col_train:
    train[col]=train[col].fillna(train[col].mode()[0])
    
for col in miss_col_test:
    test[col]=test[col].fillna(test[col].mode()[0])


train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
train.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)
test.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)


le = LabelEncoder()
for col in train.select_dtypes(include=['object']):
    train[col]=le.fit_transform(train[col])
for col in test.select_dtypes(include=['object']):
    test[col]=le.fit_transform(test[col])

train["SalePrice"] = np.log1p(train["SalePrice"])
from scipy.stats import norm, skew 
numeric_feats = train.dtypes[train.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    train[feat] = boxcox1p(train[feat], lam)

from scipy.stats import norm, skew 
numeric_feats = test.dtypes[test.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = test[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    test[feat] = boxcox1p(test[feat], lam)  

X=train.drop(["SalePrice","Id"],axis=1)
y=train["SalePrice"]
print(X)    

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X , y ,test_size = 0.1,random_state = 0)


from xgboost import XGBRegressor

model = XGBRegressor()
model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(X, y, early_stopping_rounds=5, 
             eval_set=[(x_test, y_test)], verbose=False)
acc = model.score(x_test, y_test)
pred_xgb = model.predict(x_test)  
type(x_test)
x_test.shape
print('Accuracy: ', acc)

k = 11 #number of variables for heatmap
corrmat = train.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

import pickle


# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

#-------------------debugg------------------
columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'TotalSF']


maxx=[]
for column in X:
    print(column)
    maxx.append(X[column].max())
    
print(maxx)


from scipy.special import inv_boxcox1p
    
columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'TotalSF']
values = [7.990963041593332, 1.8203341036428238, 9.125735246126716, 35.391370879389704, 0.7304631471189666, 0.7304631471189666, 1.5409627556327752, 1.5409627556327752, 0.7304631471189666, 1.8203341036428238, 1.1943176378757767, 4.137710644618417, 2.6025944687727294, 2.440268378362637, 1.8203341036428238, 2.440268378362637, 2.885846472488202, 2.750250297485029, 14.198437226858903, 14.198437226858903, 2.055641538058108, 2.440268378362637, 3.340760310539712, 3.4381104434026533, 1.5409627556327752, 13.496903957925936, 1.5409627556327752, 1.8203341036428238, 2.055641538058108, 1.5409627556327752, 1.5409627556327752, 1.5409627556327752, 2.055641538058108, 17.692270372158852, 2.055641538058108, 13.25049937800808, 14.673978647470983, 17.983824862473387, 2.055641538058108, 1.8203341036428238, 0.7304631471189666, 1.8203341036428238, 17.026675377207216, 14.283056501227003, 10.616842895097365, 17.690975636691796, 1.5409627556327752, 1.1943176378757767, 1.5409627556327752, 1.1943176378757767, 2.6025944687727294, 1.5409627556327752, 1.5409627556327752, 3.340760310539712, 2.2596737867230705, 1.5409627556327752, 1.8203341036428238, 2.055641538058108, 14.198437226858903, 1.1943176378757767, 1.8203341036428238, 13.135198556618896, 1.8203341036428238, 1.8203341036428238, 1.1943176378757767, 11.695834404033846, 10.501575267176273, 10.524981344043688, 10.312501443831783, 10.169007130464756, 11.289160096627912, 1.1943176378757767, 1.5409627556327752, 1.5409627556327752, 21.677435084678006, 3.128238685769432, 14.198437226858903, 2.6025944687727294, 2.055641538058108, 20.524700947545885]
xdd=dict(zip(columns, values))

df_s = pd.DataFrame(columns=columns)

df_s = df_s.append(xdd, ignore_index=True)
prediction = model.predict(df_s)


print(prediction)
