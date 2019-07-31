LASSO implementation (including coefficient weights) for VAR(p) models
using only covariance information.

The default method is reasonably fast and reliable, at least for small systems:
![alt text](https://raw.githubusercontent.com/RJTK/ts_lasso/master/figures/ts_lasso_convergence.png)

Momentum and line-search make a big difference:
![alt text](https://raw.githubusercontent.com/RJTK/ts_lasso/master/figures/convergence.png)

Plot of the BIC over lambda -- the default fit_VAR method will use fminsearch to try to find the maximizer.
![alt text](https://raw.githubusercontent.com/RJTK/ts_lasso/master/figures/cost_path.png)

Example regularization path:
![alt text](https://raw.githubusercontent.com/RJTK/ts_lasso/master/figures/regularization_path.png)
