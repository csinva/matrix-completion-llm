from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class MissForest:
    '''Impute each nan value with via miss forest algorithm 
    '''
    def __init__(self,n_estimators = 100):
        self.n_estimators = n_estimators
        self.fitted = False
        self.imp = IterativeImputer(estimator = RandomForestRegressor(n_estimators = n_estimators))
        
    def fit(self,X_train):
        self.imp.fit(X_train)
        self.fitted = True
    
    def transform(self,X_test):
        if self.fitted == False:
             ValueError(f'Fit MissForest before imputing!')
        return self.imp.transform(X_test)
    
    def fit_transform(self,X):
        self.imp.fit(X)
        self.fitted = True
        return self.transform(X)


if __name__=="__main__": 
    m = MissForest()
    m.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    X_test = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    X_test_imp = m.transform(X_test)
    print("printing transformed X_test")
    print(" ")
    print(X_test_imp)
    print(" ")
    print("fitting and transforming X_test")
    print(" ")
    print(m.fit_transform(X_test))