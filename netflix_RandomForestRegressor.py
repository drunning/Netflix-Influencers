import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from operator import itemgetter
import statsmodels.api as sm




# path of data used and generated by script
path = '../Data/'
dfw = pd.read_pickle(path+'dfw_final.pkl')

# load the train and test sets
X_train = pd.read_pickle(path+'X_train.pkl')
y_train = pd.read_pickle(path+'y_train.pkl')
X_test = pd.read_pickle(path+'X_test.pkl')
y_test = pd.read_pickle(path+'y_test.pkl')


# create the parameter grid to be used in GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10, 30], 'max_features': [2, 3, 5]},
  ]

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X_train, y_train)

# create a final model using GridSearchCV's best estimator
final_model = grid_search.best_estimator_



# score the model
score_mse = cross_val_score(final_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
score_rmse = np.sqrt(-score_mse)

# make final predictions on the test set
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# obtain feature importances and store top 15 in features
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances_list = []
for feature, importance in zip(dfw.columns, feature_importances):
    feature_importances_list.append((feature, importance))
feature_importances_list = sorted(feature_importances_list, key=itemgetter(1), reverse=True)
features = [x[0] for x in feature_importances_list[0:15]]

# create dataframe consisting of the top 15 features and target
df = dfw[features].copy()
df['target'] = dfw['TARGET'].values

# generate csv file for charting
df.to_csv(path+'random_forest_regressor.csv')


# create dataframe of dataset, then scale
dfs = pd.read_pickle(path+'dfw_final.pkl')
dfs.reset_index(inplace=True)
dfs_date = dfs['new_date']
dfs.drop(columns='new_date', inplace=True)
df_scaled = pd.DataFrame(data=preprocessing.scale(dfs), index=None, columns=dfs.columns)
df_s = df_scaled[features].copy()
df_s['TARGET'] = df_scaled['TARGET'].values
df_s['date'] = dfs_date.values
df_s.set_index('date', inplace=True)

# generate csv of scaled dataframe for charting
df_s.to_csv('../Data/random_forest_regressor_scaled.csv')


# create a new dataframe with all features
df = pd.read_pickle(path+'dfw_final.pkl')

# use the above dataframe to make X and y sets for OLS
X = df[features].copy()
y = df['TARGET']
X = sm.add_constant(X)

# generate OLS model to review R2 and other statistics
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
print(model.summary())
