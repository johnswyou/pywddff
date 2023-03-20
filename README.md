# pywddff

`pywddff` is a Python package to perform feature engineering using wavelet transforms, geared towards those who
work with tabular data and machine learning models to forecast time series. `pywddff` supports 2 wavelet transform methods:

1. Maximal Overlap Discrete Wavelet Transform (MODWT)
2. A Trous Discrete Wavelet Transform

We expect that the user is familiar with wavelet transforms (Discrete Wavelet Transform (DWT), MODWT, A Trous DWT, Multi-resolution Analysis, etc.)
Although not strictly required, knowledge of wavelet transforms and related concepts will make reading everything that follows below, and using `pywddff`
much easier. The gold standard reference textbook to learn about wavelet transforms is "Wavelet Methods for Time Series Analysis" by Percival and Walden (2000).

## Some background

Wavelet transforms are used in various application domains. One of the better known uses of wavelet transforms is in the field of image compression.
JPEG 2000 (JP2) is an image compression standard that was created by JPEG (Joint Photographic Experts Group) which uses the Discrete Wavelet Transform (DWT) [1].

Wavelet transforms are also used to analyze time series data. Less known, however, is that they can also be used to engineer input features, which can
subsequently be used to train a machine learning model for time series forecasting applications. Making this latter use of wavelet transforms more accessible is the main motivation behind `pywddff`.

The name `pywddff` is an abbreviation of "Python" and " Wavelet Data Driven Forecasting Framework" (WDDFF). WDDFF is a creation of (Quilty and Adamowski, 2018) [2] in which the two authors present a systematic way of incorporating wavelet transforms into machine learning powered forecasting pipelines.

`pywddff` is meant to aid in implementing WDDFF by taking care of the wavelet transform portion. See the section "Further details on WDDFF" at the bottom of this page for further details on WDDFF.

## Tutorial

`pywddff` is used to engineer new input features using either MODWT or A Trous DWT, given an existing feature set `X`.
In forecasting, `X` will be a set of explanatory variables lagged `h` time units behind the corresponding target/label value `y`,
where `h` is the forecast horizon. 

For example, suppose we want to forecast tomorrow's daily mean temperature. The following shows an
example of how we might prepare a data set given a weeks worth of daily average temperatures:

``` python
import numpy as np
from scipy.ndimage import shift

# Units: Fahrenheit or Celsius, whichever you prefer
# In daily_temperature, left is going back in time, right is going forward in time
daily_temperature = np.array([25., 24., 23., 25., 21., 29., 28.])

# Create X
historical_daily_temperature = shift(daily_temperature, 1, cval=np.NaN)
# array([ nan, 25, 24, 23, 25, 21, 29])

# Column concatenate the y and X
data_set = np.column_stack((daily_temperature, historical_daily_temperature))
# array([[25., nan],
#        [24., 25.],
#        [23., 24.],
#        [25., 23.],
#        [21., 25.],
#        [29., 21.],
#        [28., 29.]])

# Remove the first row containing nan (we will always have to remove the first h rows,
# where h is the forecast horizon)
data_set = data_set[~np.isnan(data_set).any(axis=1)]
# array([[24., 25.],
#        [23., 24.],
#        [25., 23.],
#        [21., 25.],
#        [29., 21.],
#        [28., 29.]]

# Split data_set back up into y and X
y = data_set[:, 0] # array([24., 23., 25., 21., 29., 28.])
X = data_set[:, 1] # array([25., 24., 23., 25., 21., 29.])
```

Now that we know how to construct a data set for training from a multivariate time series saved as a 1D
numpy array, let's see another, more realistic example.

```python
import numpy as np
from scipy.ndimage import shift
from pywddff.stationary_dwt import multi_stationary_dwt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Forecast horizon is 1 (use your imagination for the units)
N = 1000
y = np.random.random(N)
X = shift(y, 1, cval=np.NaN)

yX = np.column_stack((y, X))
yX = yX[~np.isnan(yX).any(axis=1)]

y = yX[:, 0]
X = yX[:, 1]

# Need to reshape X because multi_stationary_dwt expects len(X.shape) to be 2
X = X.reshape((N-1, 1))

# This is where we engineer new features, 3 new features in our case since J = 2 below
X, y = multi_stationary_dwt(X.reshape((100000, 1)), y, transform="modwt", filter="coif2", J=2, pandas_output=False)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Train machine learning model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
model.score(X_test, y_test)
```

The observant reader will note that I used `multi_stationary_dwt` before splitting the data into train and test sets.
Don't worry! This is normal practice in `pywddff` and you can rest assured that no future data was leaked into the training process.
This observation, however, brings out one key downside of feature engineering using wavelet transforms. Namely, that an entire model has to
be refit whenever new observations come in (assuming the model is in production), since we cannot apply wavelet transforms in an "online" fashion. Thus, when a new observation comes in, we must:

1. Add the new observation of input features and target/label/response to our full data set.
2. Wavelet transform each input feature for the entire data set.
3. Retrain our model on the coefficients matrix (if using the Single wavelet-based forecasting approach).

## Using `pywddff` in practice: a short guide

In practice, we expect `pywddff` to be part of a larger machine learning time pipeline. `pywddff` was built with
tabular data sets in mind. The following addresses some of the key practical details without going into concrete implementation details.
Although your pipeline will likely look different from what's shown below, hopefully you can get a sense of how `pywddff` can be incorporated
into a larger machine learning pipeline.

To create new wavelet transform based features, there are a few choices that need to be made by the user:

1. How many new features do you want? This will determine `J`, the decomposition level.
2. Which `filter` do you want to use? There are 128 available choices.
3. Which transform method do you want to use? There's 2 available: MODWT and A Trous DWT.
4. Do you want the output to be a numpy array or a pandas dataframe?

The choices for 1 to 3 above are not trivial, and is a subject of research. Grid search is usually the
most practical approach if you can afford the computational burden of retraining your model over and over again for possibly
hundreds or even thousands of times. To perform grid search, define 

1. The set of filters to try.
2. The set of decomposition levels to try.
3. The set of wavelet transforms (just 2 available, MODWT and A Trous DWT).

The cartesian product of the three sets above produces a grid of `(filter, j, transform)` combinations that you will
evaluate over (more on this shortly). A an example starting point for grid construction is

1. All 128 filters available in `pywddff`.
2. Decomposition levels `j` of 1 to 6 (inclusive).

Note that the above specifications are reasonable only if you have a sufficiently large dataset (e.g., 10 thousands rows or more) and your forecasting model
trains quickly, such as linear regression. Wavelet transform based feature engineering is NOT recommended for small datasets due to the potentially large
number of data points that will be lost due to removing boundary coefficients.

Always be mindful that `(2^max_J - 1)*(max_L - 1)` (where `max_J` is the largest decomposition level in your grid and `max_L` is the longest filter length in your grid) is the maximum number of rows that will be removed from your data set to account for boundary coefficients. This is why your data set should be of sufficient size so that the boundary coefficient rows lost is a small fraction of the full data set size. The requirement of removing boundary coefficient rows is the most significant drawback of using wavelet transforms for feature engineering.

So what does evaluating over a grid look like? Using the Single wavelet-based forecasting approach, it might look like this:

1. Define an evaluation metric to optimize. An evaluation metric determines how well a model forecasts, and is typically a function
   of two arguments: a vector of forecasts (call this `y_pred`) and a vector of true observations (call this `y_obs`).
2. For the ith combination in a grid of filter and decomposition level combinations, perform MODWT or A trous DWT on each column of the
   input feature matrix `X`. This will produce a new matrix of wavelet coefficients and scaling coefficient with `ncol*(j+1)` columns, where
   `ncol` is the number of columns in `X` and `j` is the decomposition level in the current ith grid element.
3. Partition the coefficients matrix obtained in step 2 above into train, validation and test sets.
4. Normalize the columns of the training set coefficients matrix, save the columnwise mean and standard deviation statistics computed on the training 
   coefficients matrix, and apply normalization to the validation and test set coefficients matrices using the statistics computed on the training coefficients matrix.
5. Train a machine learning model (e.g., linear regression) on the training coefficients matrix and corresponding target `y`. Make predictions on the 
   validation coefficients matrix to obtain `y_pred`.
6. Evaluate chosen evaluation metric (e.g., RMSE, R-Squared, etc.) using `y` and `y_pred`.
7. Repeat step 2 to 6 for the next grid element.
8. Return the grid element that yielded the best validation set evaluation metric value.

## Further details on WDDFF

Crucially, [2] shows that MODWT and A trous DWT are the only valid wavelet decomposition methods that can be used for forecasting time series, and that many
prior studies in the field of hydrologic time series forecasting have used wavelet transforms incorrectly by allowing future information to leak into the
training set. The authors of [2] propose WDDFF as a way of correctly (i.e., only using historical data) performing MODWT or A Trous DWT for the purpose of training downstream machine learning models for forecasting.

If you read [2], which is highly recommended, section 3.2.1.1 lists 6 different wavelet-based forecasting methods. `pywddff` currently only
supports method 1 (Single). Method 4 (Single-hybrid) can be trivially obtained by concatenating the result of `multi_stationary_dwt` to the original input feature set `X` (after removing boundary coefficients from `X` using `n_boundary_coefs`). In future versions, we may add support for the 4 other wavelet-based forecasting methods.

Only MODWT and A trous DWT are implemented in `pywddff`. Multi-Resolution Analysis (MRA) decomposition (where smooth and detail
coefficients are computed) is NOT implemented. MRA decomposition cannot be used for forecasting as it can be shown that the MRA decomposition procedure requires future data (at time steps greater than `t`) to compute coefficients at time `t`. See [2] for further details on this matter.

## Why haven't wavelet transforms seen widespread adoption in machine learning?

I believe this is due to at least 6 reasons (there's likely many more reasons):

1. Hard to distill the right information form the swath of mathematical information out there pertaining to wavelet transforms
   to synthesize into a practical procedure to be applied to machine learning powered time series forecasting problems.
2. Subtleties in the way different wavelet transforms work that make it difficult to know if one is applying a particular transform correctly to
   a time series forecasting problem without leaking future information into the training process.
3. Most of the information about wavelet transforms is very technical, often academic papers that require knowledge of specialized jargon.
4. The big downside of having to remove boundary coefficients, which in practice often means losing hundreds of rows of data from your training set.
   Historically where data was often scarce, this would have been a major drawback.
5. No clear advice on selecting the right filter and decomposition level for a given time series.
6. Cannot use wavelet transforms in an online fashion for models in production. Every new observation necessitates refitting of the production model
   (see section "Tutorial" above for further details).

Indeed, even in academic papers, authors routinely misuse wavelet transforms for time series forecasting that invalidates their application to real life
forecasting. See [2] for further details.

I hope this package can make using wavelet transforms for machine learning powered forecasting more accessible.

## Reference

[1] https://en.wikipedia.org/wiki/JPEG_2000

[2] Quilty, J., & Adamowski, J. (2018). Addressing the incorrect usage of wavelet-based hydrological and water resources forecasting models for real-world applications with best practices and a new forecasting framework. Journal of Hydrology, 563, 336–353. https://doi.org/10.1016/j.jhydrol.2018.05.003 

[3] Wavelet Methods for Time Series Analysis by Percival & Walden, (2000).
