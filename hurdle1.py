import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Read training data manually made in Excel
df = pd.read_csv("Train.csv")

# Separate features and target
x, y = df.iloc[:, 1:], df.iloc[:, :1]

# Split to get a single test record
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                    test_size=0.07, random_state=123)

# Tune XGBoost regressor
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', 
                colsample_bytree = 0.6, learning_rate = 0.45,
                max_depth = 4, n_estimators = 10)

# Fit the model
xg_reg.fit(X_train,y_train)

# Predict for test data
pred = xg_reg.predict(X_test)

# Find rmse error
rmse = np.sqrt(mse(y_test, pred))
#print("RMSE: %f" % (rmse))

# Find average rmse
err = []
for row in range(x.shape[0]):
    rec = x[row:(row+1)]
    pred = xg_reg.predict(rec)[0]
    rmse = (y[row:(row+1)].iloc[0, 0] - pred)
    #print(pred, rmse)
    err.append(abs(rmse))
print("Mean RMSE", sum(err)/len(err))

# Read Test Data Manually Made
dg = pd.read_csv("Test.csv")

# Extract features
dh = dg.iloc[:, 1:]
for row in range(dh.shape[0]):
    rec = dh[row:(row+1)]
    pred = xg_reg.predict(rec)[0]
    #print(pred)
    ## Update features
    for col in range(dg.shape[1]-2):
        if row + col < dg.shape[0]:
            dg.iloc[row+col, col] = pred
            if row + col + 1 < 6:
                dh.iloc[row+col+1, col] = pred

#print(dg)

dg.to_csv("Predicted.csv", index=False)

#print(xg_reg.feature_importances_)
xgb.plot_importance(xg_reg)

