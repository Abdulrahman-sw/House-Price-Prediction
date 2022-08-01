from statistics import mode

import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
import xgboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pickle


df = pd.read_csv("D:\Abdoo\Bachelor's degree\GP\gp prediciton model/finalISA.csv")
scaler = MinMaxScaler()
#df["size"] = scaler.fit_transform(df[["size"]])
print(df["size"])
X = df.drop(["price","Address","Title"],axis=1)
Y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=.2, random_state=5)
corr_matrix = df.corr()
print(corr_matrix)

#modelling and training step
model=LinearRegression()
model.fit(X_train, y_train)


print("Gradient Boosting Regressor")
clf = ensemble.GradientBoostingRegressor(n_estimators=300,max_depth=2,learning_rate=0.15,loss="huber")
clf.fit(X_train,y_train)
y_test,clf.predict(X_test)
print("accuracy:",clf.score(X_test,y_test))

print(clf.predict([[3,2,1,195,30.119690,31.637841,0,0,0,0,1]]))
print(clf.predict([[3,2,1,195,30.119690,31.637841,0,0,0,0,1]]))

with open ("D:\Abdoo\Bachelor's degree\GP\gp prediciton model/mymodel",'wb') as f:
    pickle.dump(clf,f)
'''
#print("Intercept",model.intercept_)
#print(pd.DataFrame(data=model.coef_,index= X_train.columns,columns=['coef']))
print("////////////////////////////")
print("Linear Regression")
model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
print("rmse: ",rmse)
print("accuracy:",model.score(X_test,y_test))
print("Decision Tree Regressor")
Dtg = DecisionTreeRegressor()
Dtg.fit(X_train,y_train)
rmse = np.sqrt(mean_squared_error(y_test,Dtg.predict(X_test)))
print("rmse: ",rmse)
print("accuracy:",Dtg.score(X_test,y_test))

print("XGB Regressor")
XGB = xgboost.XGBRegressor()
XGB.fit(X_train,y_train)
rmse = np.sqrt(mean_squared_error(y_test,XGB.predict(X_test)))
print("rmse: ",rmse)
print("accuracy:",XGB.score(X_test,y_test))

n_estimators = [100,200,300,400,500]
max_depth = [2,3,5,6,7,8]
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
loss_functions= ['squared_error', 'absolute_error', 'huber', 'quantile']
hyperparameters_grid={
"loss": loss_functions,
"n_estimators":n_estimators,
"max_depth":max_depth,
"learning_rate":learning_rate,
}
#print(X.columns)
#print(clf.predict([[3,2,1,195,30.119690,31.637841,0,0,0,0,1]]))
#print()

Grid_cv = RandomizedSearchCV(estimator=clf, param_distributions= hyperparameters_grid,return_train_score= True)
Grid_cv.fit(X_train,y_train)
print("results:",Grid_cv.cv_results_)
print("Best Params:",Grid_cv.best_params_)
print("Best Score:",Grid_cv.best_score_)

#print(clf.predict([[]]))
'''