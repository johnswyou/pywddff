# ts_wavelets

Mini Python 3.x library to perform Maximal Overlap Discrete Wavelet Transform (MODWT) on time series data

## Usage

Presently, only MODWT is implemented via the `modwt` function.

``` python
from ts_wavelets.modwt import modwt
import numpy as np

n_observations = 1000
example_time_series = np.random.random(n_observations)

# Use the Haar wavelet/scaling filter with 6 decomposition levels
modwt(example_time_series, 'haar', 6)
```

## Reference

`modwt` was written from pseudo-code provided on page 177 of Wavelet Methods for Time Series Analysis by Percival & Walden.