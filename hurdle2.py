import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor as rfr
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Data Preparation
## Read Data
df_train = pd.read_excel('TrainingData.xlsx').fillna(value=0)
df_test = pd.read_excel('TestData.xlsx').fillna(value=0)

## Ignore Categoric Variables as test cats are not in train
x_train = df_train.iloc[:, 2:-5]
x_test = df_test.iloc[:, 2:-5]
y_train = df_train.iloc[:, 1]
y_test = df_test.iloc[:, 1]

# Random Forest Regression manually tuned
def rf():
    rf_reg = rfr(n_estimators = 50, random_state = 0) 
    rf_reg.fit(x_train, y_train)
    pred = rf_reg.predict(x_test)
    rmse = np.sqrt(mse(y_test, pred))
    return rmse
    
print(rf())

# XGBoost Regression manually tuned
def xgr():
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', 
                colsample_bytree = 0.5, learning_rate = 0.38,
                max_depth = 5, n_estimators = 60, nthreads=4)
    xg_reg.fit(x_train,y_train)
    pred = xg_reg.predict(x_test)
    rmse = np.sqrt(mse(y_test, pred))
    xgb.plot_importance(xg_reg)
    return rmse

print(xgr())






