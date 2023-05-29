
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/johnswyou/pywddff/blob/6f6598fa55513b7a94974c82bb451a41c9532cbc/docs/_static/banner-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/johnswyou/pywddff/blob/6f6598fa55513b7a94974c82bb451a41c9532cbc/docs/_static/banner-light.png">
    <img alt="Shows a black logo in light color mode and a white one in dark color mode." src="https://github.com/johnswyou/pywddff/blob/6f6598fa55513b7a94974c82bb451a41c9532cbc/docs/_static/banner-light.png">
</picture>

<!-- start here -->

# pywddff

Python package to perform feature engineering using wavelet transforms, geared towards those who work with tabular data and machine learning models to forecast time series.

## Documentation

You can find the documentation for `pywddff` including API references [here](https://pywddff.readthedocs.io/en/latest/).

## About the name

`pywddff` stands for "Python Wavelet Data Driven Forecasting Framework". WDDFF is an approach to combining wavelet transform generated features and machine learning for forecasting time series. The WDDFF paper can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S0022169418303317).

## Installation

This package is not yet listed on the Python Package Index (PyPI). You can use `poetry install` to install `pywddff` by first cloning this repository and then running `poetry install` from the root directory. Please note that you must have `poetry` already installed and available from your terminal. See [here](https://python-poetry.org/docs/master/) for instructions on installing `poetry` on your system.

```bash
$ git clone https://github.com/johnswyou/pywddff.git
$ cd pywddff
$ poetry install
```

## Available wavelet/scaling filters

`pywddff` gives access to 128 (decomposition level 1) wavelet (high pass) and scaling (low pass) filters.

The following are short names for the available filters:

``` python
['bl7', 'bl9', 'bl10', 
'beyl', 
'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 
'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 
'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 
'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 
'db34', 'db35', 'db36', 'db37', 'db38', 'db39', 'db40', 'db41', 'db42', 'db43', 'db44', 'db45', 
'fk4', 'fk6', 'fk8', 'fk14', 'fk18', 'fk22', 
'han2_3', 'han3_3', 'han4_5', 'han5_5', 
'dmey', 
'mb4_2', 'mb8_2', 'mb8_3', 'mb8_4', 'mb10_3', 'mb12_3', 'mb14_3', 'mb16_3', 'mb18_3', 'mb24_3', 'mb32_3', 
'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 
'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20', 'sym21', 'sym22', 'sym23', 'sym24', 'sym25', 'sym26', 'sym27', 
'sym28', 'sym29', 'sym30', 'sym31', 'sym32', 'sym33', 'sym34', 'sym35', 'sym36', 'sym37', 'sym38', 'sym39', 'sym40', 
'sym41', 'sym42', 'sym43', 'sym44', 'sym45', 
'vaid', 
'la8', 'la10', 'la12', 'la14', 'la16', 'la18', 'la20']
```
where

- `bl` corresponds to "Best-localized Daubechies"
- `beyl` corresponds to "Beylkin"
- `coif` corresponds to "Coiflets"
- `db` corresponds to "Daubechies"
- `fk` corresponds to "Fejér-Korovkin"
- `han` corresponds to "Han linear-phase moments"
- `dmey` corresponds to "Discrete Meyer"
- `mb` corresponds to "Morris minimum-bandwidth"
- `sym` corresponds to "Symlets"
- `vaid` corresponds to "Vaidyanathan"
- `la` corresponds to "Least asymmetric"

NOTE: The `db1` is the Haar wavelet.

### Source of data

The `wfilters` function from MATALB R2022b's wavelet toolbox provided 121 of the 128 wavelet (scaling) filters available. The R package `wavelets` provided the remaining 7 least asymmetric
wavelet (scaling) filters.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`pywddff` was created by John You. It is licensed under the terms of the MIT license.

## Credits

`pywddff` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

<!-- end here -->
