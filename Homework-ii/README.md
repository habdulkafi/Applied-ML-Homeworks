


# Predicting Rent 


The model is built using the following features selected from the data set:

    boro : Borough 
        1=Bronx
        2=Brooklyn
        3=Manhattan
        4=Queens
        5=Staten Island
    sc23 : Condition of Building (Observation)
        1=Dilapidated
        2=Sound
        3=Deteriorating
        8=Not reported            recoded as nan
    sc24 : Any Buildings with Broken or Boarded-up Windows (Observation)
        1=Yes
        2=No
        8=Not reported            recoded as nan
    sc36 : Wheelchair accessibility - Street Entry and Inner Lobby Entry
        1=Accessible
        2=Inaccessible
        3=Unable to observe       recoded as nan
        8=Not reported            recoded as nan
    sc38 : Wheelchair accessibility - Residential Unit Entrance 
        1=Accessible
        2=Inaccessible
        3=Unable to observe       recoded as nan
        8=Not reported            recoded as nan
    sc54 : First Occupants of Unit
        1=Yes, first occupants
        2=No, previously occupied
        3=Don't know              recoded as nan
        8=Not reported            recoded as nan
    sc114 : Condo/Coop Status 
        1=No
        2=Yes, a condominium
        3=Yes, a cooperative
        4=Don't know              recoded as nan
    uf48 : Number of Units in Building
        01=1 unit without business
        02=1 unit with business
        03=2 units without business
        04=2 units with business
        05=3 units
        06=4 units
        07=5 units
        08=6 to 9 units
        09=10 to 12 units
        10=13 to 19 units
        11=20 to 49 units
        12=50 to 99 units
        13=100 units or more
    uf11 : Stories in Building
        01=1 to 2 stories
        02=3 stories
        03=4 stories
        04=5 stories
        05=6 to 10 stories
        06=11 to 20 stories
        07=21 stories or more
    sc149 : Passenger Elevator in Building
        1=Yes
        2=No
    sc150 : Number of Rooms 
        1=1 room
        ..
        8=8 rooms or more
    sc151 : Number of Bedrooms
        01=No bedroom
        ..
        09=8 bedrooms or more
    sc158 : Type of Heating Fuel
        1=Fuel oil
        2=Utility gas
        3=Electricity
        4=Other fuel (including CON ED steam)
    sc197 : Functioning Air Conditioning
        1=Yes, central air conditioning
        2=Yes, one or more window air conditioners
        3=No
        4=Don't know/Not sure     recoded as nan
        8=Not reported            recoded as nan
    sc198 : Carbon Monoxide Detector
        1=Yes
        2=No
        8=Not reported            recoded as nan
    sc188 : Presence of Mice and Rats
        1=Yes
        2=No
        8=Not reported            recoded as nan
    sc190 : Cracks or Holes in Interior Walls or Ceiling 
        1=Yes
        2=No
        8=Not reported            recoded as nan
    sc194 : Water Leakage Inside Apartment (House)
        1=Yes
        2=No
        8=Not reported            recoded as nan
    uf23 : Year Built Recode
        01=2000 or later
        02=1990 to 1999
        03=1980 to 1989
        04=1974 to 1979
        05=1960 to 1973
        06=1947 to 1959
        07=1930 to 1946
        08=1920 to 1929
        09=1901 to 1919
        10=1900 and earlier


The Response variable was chose as:

    uf17 : Monthly Contract Rent
        00001=(Dollar amount)
        ..
        ..
        05500=$5,500 (topcode amount)
        07999=Mean amount above topcode
        99999=Not applicable              deleted from data set

The missing values in each feature were reencoded as `np.nan`.


## Choosing the right model

I tried many options for preprocessing, feature selection, and model selection.  For imputation, I tried MICE from `fancyimpute`, but it didn't perform that much better than `mean` from the `Impute` class in sklearn, so I stuck with `Impute`.  I also tried scaling and not scaling as well as including or not including polynomial feature expansion.  The best I found was scaling with polynomial features.  I also tried PCA for dimensionality reduction (with and without polynomial expansion), but it always had a negative effect on the model's R^2 value.  

For model selection, I tried Ridge, Lasso, LinearRegression, SVR, and LinearSVR.  I did grid search on all of the paramters in each of the models with the same preprocessing pipeline (Lasso to foreverrr), and concluded that Ridge was the best model to fit.  

In the code submitted, I did not include the full search for the best model - I only included the Grid Search over the different Imputation strategies, the polynomial features bias, and the ridge alpha.  Most of my work was done in a few Jupyter notebooks, so going back and extracting all that model selection code would have been a major hassle.  

I made sure to only choose features that did not rely on current residents only.  I believe this eventually made my model result in a smaller R^2 value but reducing the influence of those useless variables on the outcome was more important to me than fitting a perfect model.  

## Cross-Validation

This is what I did for the final hyper-parameter selection.  I created an 80/20 split of the data (say `train`/`test`).  I then took `train` and performed GridSearchCV with 5 fold cross-validation for the parameters.  This means the hyper-parameters were chosen by *only* looking at the `train`ing set without leaking information from the `test` set.  Then, when I found the correct choices for the hyper-parmeters, I used them to do a 5 fold cross-validation on the *entire* data set (aka train 5 models on 5 folds, and average the R^2 value). 

## R^2

The 5 fold cross-validated R^2 value I ended up had a mean of around 0.30 with a standard deviation of about 0.02.  I did not set a seed, so each run will have different exact values for the mean (but still around .30) and also pretty different standard deviations (resulting from having only 5 folds).






-------------------------



# ORIGINAL TASK INSTRUCTIONS:

The repository should contain code to download the data and the code to process it, as well build and validate the model. Use travis to run your experiment and write a test that ensures the outcome is the actual error you report (Say, if you claim your algorithm is 90% accurate, write a test that checks that the score returned by your model is at least 90%).
 
We recommend that you fork the homework repository and run Travis on your own repository - this way you don’t have to wait for other students submissions to finish on travis.
Task 1
A real estate agent wants to estimate the market rate for some apartments in NYC that just went into the market again. They want to use the data posted by the Census in 2014:
https://www.census.gov/housing/nychvs/data/2014/userinfo2.html
(The data is at https://www.census.gov/housing/nychvs/data/2014/uf_14_occ_web_b.txt)
You can find a parsed version on figshare: https://ndownloader.figshare.com/files/7586326
 
 
Create and validate a machine learning approach to predict the monthly rent of an apartment.
Make sure to only use features that apply to pricing an apartment that is not currently rented. You can make the simplifying assumption that the market doesn’t increase, so the rent for a new tenant would be the same as for the current tenant.
 
Explain how you validated your model and why.
 
Report the test error using R^2 and any other metric you deem appropriate.
 
Limit yourself to linear models for the prediction (though feature engineering, feature selection and imputation methods are allowed). Ensembles of models are not allowed.
 
There should be a file “homework2_rent.py” with a function “score_rent” that returns the R^2 and a function “predict_rent” that returns your test data, the true labels and your predicted labels (all as numpy arrays).
The tests should be in a separate file called ``test_rent.py” with a function called “test_rent” that checks the R^2 returned by score_rent to be as least as good as your expected outcome.

