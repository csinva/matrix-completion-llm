import numpy as np
from sklearn.impute import SimpleImputer

class MedianImputer:
    def fit_transform(self, m: np.ndarray):
        '''Impute each nan value with the median of its column
        '''
        imp = SimpleImputer(missing_values = np.nan, strategy = 'median')
        m_imputed = imp.fit_transform(m)
        return m_imputed
      

if __name__=="__main__": 
    X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
    m = MedianImputer()
    print(m.fit_transform(X_test))

