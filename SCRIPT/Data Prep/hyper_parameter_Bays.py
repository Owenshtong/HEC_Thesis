##################################################
## Compute the hyperparameters for bayesian OLS ##
##           Prior variance re-computation      ##
##################################################

import numpy as np
import pandas as pd
import SCRIPT.Interpolator as itp

# Read Options
options = pd.read_csv("OUTPUT/OP_MOD150101_230228.csv")

# Index days
Days = options.date.unique()

# ATM option IV by interpolation
IV_ATM_1year = options.groupby("date").apply(
    itp.Interpolate_IV
)
IV_ATM_1month = options.groupby("date").apply(lambda x:
    itp.Interpolate_IV(x, maturity = 20/252)
)
IV_ATM_04 = options.groupby("date").apply(lambda x:
    itp.Interpolate_IV(x, log_moneyness=0.4, maturity = 20/252)
)

slope = (IV_ATM_1year - IV_ATM_1month) / np.exp(-4/12)

moneyness_slope_prox = IV_ATM_1month - IV_ATM_04

# Prior var
beta1_prior_var = np.var(IV_ATM_1year)
beta2_prior_var = np.var(slope)
beta3_prior_var = np.var(moneyness_slope_prox)
beta5_prior_var = 1 * 1e-4

beta_prior = [beta1_prior_var,beta2_prior_var,beta3_prior_var, beta5_prior_var]
