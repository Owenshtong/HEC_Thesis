###################################
########## Main Script ############
###################################

import pandas as pd
import matplotlib.pyplot as plt

from SCRIPT.options import option_cls
from SCRIPT.var.helper import rolling_forecast as rf, restore_beta as rb


### Preparation ###

# Exg vars
UMC = pd.read_csv("OUTPUT/UMC.csv", index_col=0, parse_dates=True).dropna()
MCCC = pd.read_csv("OUTPUT/MCCC_agg_daily.csv", index_col=0, parse_dates=True).dropna()


# Options data
cvx = option_cls.option("CVX", 102968)  # Go to WRDS for the id
cvx.betas_d1.index = pd.to_datetime(cvx.betas_d1.index)
cvx.betas.index = pd.to_datetime(cvx.betas.index)

ibm = option_cls.option("IBM", 106276)  # Go to WRDS for the id
ibm.betas_d1.index = pd.to_datetime(ibm.betas_d1.index)
ibm.betas.index = pd.to_datetime(ibm.betas.index)


# rolling forecast
ticker = cvx
n = 3000
burnin = 2600
window = 100
lag = 3
period = 1656

F_wo_B, F_w_B, F_wo, F_w, realized, MSE = rf(ticker.betas_d1, UMC, 3, window, n, burnin, period, pi = [1, 1, 1], own_lag_prior_mean = 0)


for i in ibm.betas.columns:
    fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
    ax.plot(ibm.betas.loc[rb(F_w_B, ibm.betas).index].loc[:,i].iloc[-500:,], label = "realized beta")
    ax.plot(rb(F_wo_B, ibm.betas).loc[:,i].iloc[-500:,], label = "without UMC Bay")
    ax.plot(rb(F_w_B, ibm.betas).loc[:, i].iloc[-500:,], label="with UMC Bay")
    # ax.plot(rb(F_wo, ibm.betas).loc[:, i], label="without UMC")
    # ax.plot(rb(F_w, ibm.betas).loc[:, i], label="with UMC")


    ax.legend()
    plt.show()

