from sklearn.impute import KNNImputer as KNN
import numpy as np

class KNNImputer:
    '''
    Simple wrapper around kNN
    '''        
    
    def fit_transform(self, m: np.ndarray,k = 3):
        '''Impute each nan value with via nearest neighbors
        '''
        imp = KNN(n_neighbors = k,weights = 'uniform')
        m_imputed = imp.fit_transform(m)
        return m_imputed
      

if __name__=="__main__": 
    X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
    m = KNNImputer()
    print(m.fit_transform(X,2))
