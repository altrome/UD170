import numpy as np
import pandas
from sklearn.linear_model import SGDRegressor

def normalize_features(features):
    ''' 
    Returns the means and standard deviations of the given features, along with a normalized feature
    matrix.
    ''' 
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features
def linear_regression(features, values):
    """
    Performs linear regression given a dataset with an arbitrary number of features.
    'features' is the input data points (or the X's) and 'values' is the output data points
    (or the Y's).
    
    Returns the intercept and the parameters, that is, the optimal values of theta.
    
    """

    model = SGDRegressor()
    model.fit(normalize_features(features), values)
    return intercept, params


