import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import statsmodels.api as sm


# read the data file into df
df = pd.read_pickle('../Data/dfw_final.pkl')

# make a new dataframe to feed into RFE, exclude the target
dfw = df['TARGET'].values
dfw = df.drop(columns='TARGET')
dfw_target = df['TARGET'].values

# create an estimator for RFE, then create an RFE object for 15 features, fit to data
estimator = SVR(kernel="linear")
selector = RFE(estimator, 15, step=0.10)
selector = selector.fit(dfw, dfw_target)

# create a features list and store in features_selected
features = []
for feature, selected in zip(dfw.columns, selector.get_support()):
    features.append((feature, selected))
features_selected = [x[0] for x in features if x[1]==True]



# load train and test sets
X_train = pd.read_pickle('../Data/X_train.pkl')
y_train = pd.read_pickle('../Data/y_train.pkl')
X_test = pd.read_pickle('../Data/X_test.pkl')
y_test = pd.read_pickle('../Data/y_test.pkl')

# but only use the features_selected
X_train = X_train[features_selected].copy()
X_test = X_test[features_selected].copy()


# create parameter grid to use in GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30, 40, 60], 'max_features': [2, 4, 5, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10, 30, 40], 'max_features': [2, 3, 5]},
  ]

# create a RandomForestRegressor object to feed to GridSearchCV
forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
# fit data
grid_search.fit(X_train, y_train)

# save the final model using GridSearchCV's best estimator
final_model = grid_search.best_estimator_

# get final predictions on the test set
final_predictions = final_model.predict(X_test)

# score the final predictions
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# create X consisting of the features selected and y consisting of TARGET
# and run through OLS model to obtain R2 and other statistics

X = df[features_selected].copy()
y = df['TARGET']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)
print(model.summary())
