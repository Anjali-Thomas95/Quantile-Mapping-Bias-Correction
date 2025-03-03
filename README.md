# Quantile-Mapping-Bias-Correction for Ensemble Data

This repository provides a Python implementation of quantile mapping bias correction for ensemble datasets. The method is designed to correct biases in modeled datasets by aligning their statistical distributions with observed data.

# Methodology
1. Ensemble Data Arrangement: The ensemble data is structured so that, for each time step, multiple ensemble members are available (e.g., 3000 values for a given day).
2. Quantile Calculation: The modeled ensemble values and observed dataset are both sorted into quantiles at each grid point.
3. Bias Correction: The difference between corresponding quantiles of the modeled and observed datasets is computed as:
   Correction Value = Modeled Value (q-th percentile)−Observed Value (q-th percentile) 
   The correction is then applied to adjust the modeled data.
4. Grid-Point-Based Correction: This process is performed independently at each grid point to ensure spatial consistency.
   
This approach is particularly useful for applications such as climate model post-processing, where reducing biases in precipitation, temperature, or other meteorological variables is essential for reliable simulations.
