# Predicting segment speed

This is a narrative commentary on trying to predict the speed of a GPX segment

## Basic regression

For a baseline approach, I started with linear regression. My predictors were slope and elevation. The results were not great:

```
speed = 5.0801 + -0.0155*slope + -0.000288*elevation
Test RMSE: 0.838 mph
Test MAE: 0.664 mph
Test R^2: 0.533
```
<img width="1260" height="1260" alt="image" src="https://github.com/user-attachments/assets/945ff45c-02bc-4e5c-8ba8-da9325b98925" />

Next, I added a few more predictors: uphill_slope (max(slope, 0)), downhill_slope (max(-slope, 0)), abs_slope, slope^2, and elevation * uphill_slope, to capture effects of climbing at altitude. 

Here is the new equation: 
```
speed_mph =
  5.4301
  - 0.033098 * slope
  - 0.000258 * elevation
  - 0.102784 * abs_slope
  - 0.116647 * uphill_slope
  - 0.029386 * downhill_slope
  + 0.001465 * slope_squared
  + 0.000012 * elevation_x_uphill_slope
```
and here are the results:
```
Test RMSE: 0.794 mph
Test MAE: 0.597 mph
Test R^2: 0.581
```

<img width="1260" height="1260" alt="image" src="https://github.com/user-attachments/assets/1c763980-d844-481e-88af-fc3512710b97" />

Mild improvement! The model at least is now making predictions out to 4 mph, while it was more conservative before.
