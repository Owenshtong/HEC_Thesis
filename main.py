###################################
########## Main Script ############
###################################
import matplotlib.pyplot as plt
import pandas as pd
from SCRIPT.options import option_cls
import SCRIPT.var.helper as helper
from SCRIPT.var.helper import rolling_forecast as rf


### 1. Preparation ###

# Exg vars
UMC = pd.read_csv("OUTPUT/UMC.csv", index_col=0, parse_dates=True).dropna()

# Options data
cvx = option_cls.option("CVX", 102968)  # Go to WRDS for the id
cvx.betas_d1.index = pd.to_datetime(cvx.betas_d1.index)
cvx.betas.index = pd.to_datetime(cvx.betas.index)

nee = option_cls.option("NEE", 104560)  # Go to WRDS for the id
nee.betas_d1.index = pd.to_datetime(nee.betas_d1.index)
nee.betas.index = pd.to_datetime(nee.betas.index)


### 2. Hyper Parameter Calibration ###

# We calibrate the pi1 and pi 2 for the two models
# Pi3 is set to be 100 by some experiments
# Results should be chosen manually and plugin to the rolling forecast function




# rolling forecast
ticker = cvx
n = 1000
burnin = 800
window = 100
lag = 3
period = 20

# Calibration of
F_wo_B, F_w_B, F_wo, F_w, realized = rf(ticker.betas_d1, UMC, 3, window, n, burnin,
                                        [1.425, 0.00369, 10],
                                        period,
                                        own_lag_prior_mean = 0)



b_true = helper.restore_beta(realized, ticker.betas)
b_hat = helper.restore_beta(F_wo_B, ticker.betas)
b_hat_UMC = helper.restore_beta(F_w_B, ticker.betas)

for i in F_wo_B.columns:
    fig, ax = plt.subplots(figsize=(50, 10), dpi=200)
    ax.plot(b_true[i], label=i + "_Realized")
    ax.plot(b_hat[i], label=i)
    ax.plot(b_hat_UMC[i], label = i + "_UMC")
    plt.margins(x=0)
    plt.legend()
    ax.set_xticks(ax.get_xticks()[::200])
    plt.show()

for i in F_wo_B.columns:
    fig, ax = plt.subplots(figsize=(50, 10), dpi=200)
    ax.plot(realized[i], label=i + "_Realized")
    ax.plot(F_wo_B[i], label=i)
    ax.plot(F_w_B[i], label = i + "_UMC")
    plt.margins(x=0)
    plt.legend()
    ax.set_xticks(ax.get_xticks()[::200])
    plt.show()

