from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression, Lasso
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import Imputer, StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

def download_clean_data():
    """Downloads the data, removes response-less rows, and encodes missing as nan

    Parameters
    ----------
    None
    
    Returns
    -------
    X : ndarray
        Cleaned up predictor variables of shape (n, d)
    y : ndarray
        Response variable of shape (n, )

    """
    url = "https://ndownloader.figshare.com/files/7586326"

    # Download data
    df = pd.read_csv(url)

    # Remove rows without response
    df = df[df['uf17'] != 99999]

    # Select important columns from the data set:
    # boro : Borough 
        # 1=Bronx
        # 2=Brooklyn
        # 3=Manhattan
        # 4=Queens
        # 5=Staten Island
    # sc23 : Condition of Building (Observation)
        # 1=Dilapidated
        # 2=Sound
        # 3=Deteriorating
        # 8=Not reported            recoded as nan
    # sc24 : Any Buildings with Broken or Boarded-up Windows (Observation)
        # 1=Yes
        # 2=No
        # 8=Not reported            recoded as nan
    # sc36 : Wheelchair accessibility - Street Entry and Inner Lobby Entry
        # 1=Accessible
        # 2=Inaccessible
        # 3=Unable to observe       recoded as nan
        # 8=Not reported            recoded as nan
    # sc38 : Wheelchair accessibility - Residential Unit Entrance 
        # 1=Accessible
        # 2=Inaccessible
        # 3=Unable to observe       recoded as nan
        # 8=Not reported            recoded as nan
    # sc54 : First Occupants of Unit
        # 1=Yes, first occupants
        # 2=No, previously occupied
        # 3=Don't know              recoded as nan
        # 8=Not reported            recoded as nan
    # sc114 : Condo/Coop Status 
        # 1=No
        # 2=Yes, a condominium
        # 3=Yes, a cooperative
        # 4=Don't know              recoded as nan
    # uf48 : Number of Units in Building
        # 01=1 unit without business
        # 02=1 unit with business
        # 03=2 units without business
        # 04=2 units with business
        # 05=3 units
        # 06=4 units
        # 07=5 units
        # 08=6 to 9 units
        # 09=10 to 12 units
        # 10=13 to 19 units
        # 11=20 to 49 units
        # 12=50 to 99 units
        # 13=100 units or more
    # uf11 : Stories in Building
        # 01=1 to 2 stories
        # 02=3 stories
        # 03=4 stories
        # 04=5 stories
        # 05=6 to 10 stories
        # 06=11 to 20 stories
        # 07=21 stories or more
    # sc149 : Passenger Elevator in Building
        # 1=Yes
        # 2=No
    # sc150 : Number of Rooms 
        # 1=1 room
        # ..
        # 8=8 rooms or more
    # sc151 : Number of Bedrooms
        # 01=No bedroom
        # ..
        # 09=8 bedrooms or more
    # sc158 : Type of Heating Fuel
        # 1=Fuel oil
        # 2=Utility gas
        # 3=Electricity
        # 4=Other fuel (including CON ED steam)
    # sc197 : Functioning Air Conditioning
        # 1=Yes, central air conditioning
        # 2=Yes, one or more window air conditioners
        # 3=No
        # 4=Don't know/Not sure     recoded as nan
        # 8=Not reported            recoded as nan
    # sc198 : Carbon Monoxide Detector
        # 1=Yes
        # 2=No
        # 8=Not reported            recoded as nan
    # sc188 : Presence of Mice and Rats
        # 1=Yes
        # 2=No
        # 8=Not reported            recoded as nan
    # sc190 : Cracks or Holes in Interior Walls or Ceiling 
        # 1=Yes
        # 2=No
        # 8=Not reported            recoded as nan
    # sc194 : Water Leakage Inside Apartment (House)
        # 1=Yes
        # 2=No
        # 8=Not reported            recoded as nan
    # uf23 : Year Built Recode
        # 01=2000 or later
        # 02=1990 to 1999
        # 03=1980 to 1989
        # 04=1974 to 1979
        # 05=1960 to 1973
        # 06=1947 to 1959
        # 07=1930 to 1946
        # 08=1920 to 1929
        # 09=1901 to 1919
        # 10=1900 and earlier
    # uf17 : Monthly Contract Rent
        # 00001=(Dollar amount)
        # ..
        # ..
        # 05500=$5,500 (topcode amount)
        # 07999=Mean amount above topcode
        # 99999=Not applicable              deleted from data set
    df = df[['boro','sc23','sc24', 'sc36', 'sc38', 'sc54', 
             'sc114', 'uf48', 'uf11', 'sc149', 'sc150', 'sc151', 
             'sc158', 'sc197', 'sc198', 'sc188', 'sc190', 'sc194', 
             'uf23', 'uf17']]
    
    df['sc23'] = df['sc23'].replace(8, np.nan)

    df['sc24'] = df['sc24'].replace(8, np.nan)

    df['sc36'] = df['sc36'].replace(8, np.nan)
    df['sc36'] = df['sc36'].replace(3, np.nan)

    df['sc38'] = df['sc38'].replace(8, np.nan)
    df['sc38'] = df['sc38'].replace(3, np.nan)

    df['sc54'] = df['sc54'].replace(8, np.nan)
    df['sc54'] = df['sc54'].replace(3, np.nan)

    df['sc114'] = df['sc114'].replace(4, np.nan)

    df['sc197'] = df['sc197'].replace(8, np.nan)
    df['sc197'] = df['sc197'].replace(4, np.nan)

    df['sc198'] = df['sc198'].replace(8, np.nan)

    df['sc188'] = df['sc188'].replace(8, np.nan)

    df['sc190'] = df['sc190'].replace(8, np.nan)

    df['sc194'] = df['sc194'].replace(8, np.nan)

    X = df[['boro','sc23','sc24', 'sc36', 'sc38', 
            'sc54', 'sc114', 'uf48', 'uf11', 'sc149', 
            'sc150', 'sc151', 'sc158', 'sc197', 'sc198', 
            'sc188', 'sc190', 'sc194', 'uf23']].as_matrix()
    y = df['uf17'].values
    return (X, y)


def find_hyper_params(X, y):
    """Does a Grid Search for the best hyper-parameters of the model

    Parameters
    ----------
    X : ndarray 
        A multi-dimensional array of shape (n, d)

    y : ndarray
        A single dimensional array of shape (n, )
    
    Returns
    -------
    params : dictionary
        A dictionary with the best parameters based on a CV Grid Search

    """
    # Create grid of various hyper-parameters 
    parameters = {"imputer__strategy": ("mean", "median", "most_frequent"), 
              "polynomialfeatures__include_bias": (True, False),
              "ridge__alpha": np.logspace(-3, 3, 7)
             }
    # Set up pipeline and grid search that imputes the data, scales it,  
    #   # adds poly features, and performs Ridge Regression 
    pipe = make_pipeline(Imputer(), StandardScaler(), PolynomialFeatures(), Ridge())
    grid = GridSearchCV(pipe, param_grid=parameters, cv=5, n_jobs=8)
    # Search the grid for the best parameters for the data
    grid.fit(X, y)
    return grid.best_params_


def score_rent(X, y):
    """Builds a linear model and returns a Cross-validated R^2 value

    Parameters
    ----------
    X : ndarray 
        A multi-dimensional array of shape (n, d)

    y : ndarray
        A single dimensional array of shape (n, )
    
    Returns
    -------
    r2 : float
        The R^2 value of the linear model

    """
    # randomly shuffle the data
    new_X, new_y = shuffle(X, y)
    # do an 80/20 train/test split
    train_X = new_X[:int(.8*new_X.shape[0])]
    test_X = new_X[int(.8*new_X.shape[0]):]
    train_y = new_y[:int(.8*new_X.shape[0])]
    test_y = new_y[int(.8*new_X.shape[0]):]
    # Do a grid search for the best params
    # best_parameters = find_hyper_params(train_X, train_y)
    #### Just for the script, I hard-coded the best parameters I found with grid search CV
    best_parameters = {'imputer__strategy': 'mean', 
                       'polynomialfeatures__include_bias': False, 
                       'ridge__alpha': 100.0}
    # Set up pipeline
    pipe = make_pipeline(Imputer(), StandardScaler(), PolynomialFeatures(), Ridge())
    # set the parameters found previously
    pipe.set_params(**best_parameters)
    # compute R^2 5 times based on 5 different train/test splits
    scores = cross_val_score(pipe, new_X, new_y, cv=5, n_jobs = 5)
    r2 = np.mean(scores)
    print(scores)
    print("CV std: \t", np.std(scores))
    return r2



def predict_rent(X, y):
    """Predicts values for 20% test set of the data

    Parameters
    ----------
    X : ndarray 
        A multi-dimensional array of shape (n, d)

    y : ndarray
        A single dimensional array of shape (n, )
    
    Returns
    -------
    test_X_df : DataFrame
        The test set used for the prediction
    test_y : ndarray
        The actual labels for the test set
    preds : ndarray
        The predicted labels for the test set

    """
    # randomly shuffle the data
    new_X, new_y = shuffle(X, y)
    # do an 80/20 train/test split
    train_X = new_X[:int(.8*new_X.shape[0])]
    test_X = new_X[int(.8*new_X.shape[0]):]
    train_y = new_y[:int(.8*new_X.shape[0])]
    test_y = new_y[int(.8*new_X.shape[0]):]
    # Do a grid search for the best params
    # best_parameters = find_hyper_params(train_X, train_y)
    #### Just for the script, I hard-coded the best parameters I found with grid search CV
    best_parameters = {'imputer__strategy': 'mean', 
                       'polynomialfeatures__include_bias': False, 
                       'ridge__alpha': 100.0}
    # Set up pipeline
    pipe = make_pipeline(Imputer(), StandardScaler(), PolynomialFeatures(), Ridge())
    # set the parameters found previously
    pipe.set_params(**best_parameters)
    pipe.fit(train_X, train_y)
    print("Non-CV score: \t", pipe.score(test_X, test_y))
    preds = pipe.predict(test_X)
    test_X_df = pd.DataFrame(test_X,columns=['boro','sc23','sc24', 'sc36', 'sc38', 
        'sc54', 'sc114', 'uf48', 'uf11', 'sc149', 'sc150', 'sc151', 'sc158', 
        'sc197', 'sc198', 'sc188', 'sc190', 'sc194', 'uf23'])
    return (test_X_df, test_y, preds)



if __name__ == '__main__':
    X, y = download_clean_data()
    r2 = score_rent(X, y)
    print("CV score: \t", r2)
    predict_rent(X,y)



