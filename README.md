# deriving_wetbulb
There are many methods for calculating the wet bulb temperature like one-third rule and Stull's empirical formula, but the most accurate of them all is Normand's rule. The drawback to using Normand's rule is that it requires an iterative process to converge onto the solution and cannot be applied in a vectorized way which can lead to significant compute time if the array length is long. Here I explore deriving wet bulb temperatures using both a linear regression and a deep learning method, benchmarking against the other methods--treating Normand's rule as ground truth. I will be using NCEP Reanalysis to train and test my model and KMRY station timeseries for validation.

Reference Links
Two Simple and Accurate Approximations for Wet-Bulb Temperature in Moist Conditions, with Forecasting Applications (Knox, 2017) :
https://journals.ametsoc.org/doi/10.1175/BAMS-D-16-0246.1

Wet-Bulb Temperature from Relative Humidity and Air Temperature (Stull, 2011):
https://journals.ametsoc.org/doi/full/10.1175/JAMC-D-11-0143.1

MetPy's implementation of Normand's rule:
https://unidata.github.io/MetPy/latest/_modules/metpy/calc/thermo.html#wet_bulb_temperature

The half method and one-third rule tend to overestimate the wet bulb temperature within the Tropics; lreg (linear regression) has similar biases with the half method, but less; Stull's method does well across except the poles, the keras (deep learning) model exhibits almost no bias across.
![Animation](https://github.com/ahuang11/deriving_wetbulb/blob/master/bias_maps.gif)
